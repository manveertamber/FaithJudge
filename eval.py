import argparse
import json
import os
import math
import pandas as pd
import time
import re
import logging
from tqdm import tqdm
from openai import OpenAI

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TEMPLATE = '''Your task is to evaluate a summary and determine whether it contains hallucinations, such as claims, details, or contradictions that are not supported by the source article.

You will be provided with the source article and annotated examples of summaries of the article. These examples contain hallucinations independently identified by human annotators, categorized as Benign, Unwanted, or Questionable, along with brief explanations. Use these annotated examples to guide your analysis.

Hallucination Categories:
• Benign: Information not present in the article but reasonable, supported by world knowledge, common sense, or logical reasoning, thus acceptable to readers.
• Unwanted: Problematic hallucinations, including contradictions, misrepresentations, or unsupported details.
• Questionable: Possible hallucinations open to interpretation, where annotators might reasonably disagree.

Your task is to provide a final classification of the summary as follows:
• Consistent: Summary contains no hallucinations.
• Inconsistent: Summary contains any hallucinations, whether Benign or Unwanted.

------

Source Article:
"{}"

------

Annotated Examples:

{}

------

Summary to Evaluate:
"{}"

------

Provide your reasoning first, then clearly state your final classification (Inconsistent if hallucinations of any kind are present, Consistent otherwise) at the end in this format:
"Final classification: [Inconsistent / Consistent]"
If the provided text is unrelated to the article and does not summarize it, state clearly: "Final classification: Invalid".
Do not include any additional text after the final classification.
'''


def extract_final_classification(text):
    PREDICTION_CATEGORIES = ["Inconsistent", "Consistent", "Invalid"]
    category_regex = re.compile(r"\b(" + "|".join(PREDICTION_CATEGORIES) + r")\b", re.IGNORECASE)
    all_matches = [(match.start(), match.group(0)) for match in category_regex.finditer(text)]
    
    if not all_matches:
        return None

    valid_matches = []
    for start_idx, matched_category in all_matches:
        # Look back ~15 characters for negation indicators
        preceding_text = text[max(0, start_idx - 15):start_idx]
        
        # 1) Check for "non <category>" or "non-<category>"
        snippet_around = text[max(0, start_idx - 4): start_idx + len(matched_category)]
        if re.search(rf"\bnon[\s-]+{re.escape(matched_category)}\b", snippet_around, re.IGNORECASE):
            continue
        
        # 2) Check if "not" appears among the preceding words
        preceding_words = preceding_text.split()
        last_two_words = [w.lower() for w in preceding_words[-2:]]
        if "not" in last_two_words:
            continue
        
        valid_matches.append((start_idx, matched_category))
    
    # If no valid (non-negated) match was found, we fallback to the last mention overall
    if not valid_matches:
        valid_matches = all_matches

    # Return the last match, capitalized
    final_category = valid_matches[-1][1].capitalize()
        
    return final_category


def load_summary(base_dir, model_identifier):
    """
    Load summaries from the newest CSV for a single model. The model_identifier should be in the format 'organization/model'.
    The CSV is expected to have columns: 'source_id', 'source', and 'summary'.
    Returns a dict mapping source_id to a tuple (source, summary).
    """
    try:
        organization, model_name = model_identifier.split('/', 1)
    except ValueError:
        logging.error(f"Invalid model identifier: {model_identifier}. Expected format 'organization/model'.")
        return {}

    model_path = os.path.join(base_dir, organization, model_name)
    logging.info(f"Checking directory: {model_path}")

    if not os.path.exists(model_path):
        logging.error(f"Directory not found: {model_path}")
        return {}

    csv_files = [f for f in os.listdir(model_path) if f.endswith('.csv')]
    if not csv_files:
        logging.error(f"No CSV files found for {organization}/{model_name}")
        return {}

    newest_csv = max(csv_files, key=lambda f: os.path.getctime(os.path.join(model_path, f)))
    csv_path = os.path.join(model_path, newest_csv)
    logging.info(f"Using CSV file: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read CSV at {csv_path}: {e}")
        return {}

    required_columns = {'source_id', 'source', 'summary'}
    if not required_columns.issubset(set(df.columns)):
        logging.error(f"Missing required columns in {csv_path}. Required columns: {required_columns}")
        return {}

    summaries = {}
    for _, row in df.iterrows():
        source_id = str(row['source_id'])
        source = row['source']
        summary = row['summary']
        # Only store valid summaries
        if summary is None or (isinstance(summary, float) and math.isnan(summary)) or (isinstance(summary, str) and summary.strip() == ""):
            summary = ""
        summaries[source_id] = (source, summary)
    return summaries


def load_eval_data(jsonl_file_path):
    """
    Load evaluation data from a JSONL file and return a mapping of normalized source to aggregated annotations.
    Each JSON line is expected to have: source_id, source, summary, and annotations.
    We group by the stripped source article.
    """
    eval_data = {}
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_data = json.loads(line)
                source_id = str(line_data['source_id']).strip()
                eval_data[source_id] = {
                    'source': line_data['source'].strip(),
                    'annotated_summaries': [summary for summary in line_data['summaries']],
                }
    except Exception as e:
        logging.error(f"Error reading evaluation data: {e}")
    return eval_data


def build_prompt_examples(annotated_summaries, current_summary):
    prompt_examples = ""
    current_example_num = 1
    current_summary_clean = str(current_summary).replace('\n', ' ').strip()

    for annotated_summary in annotated_summaries:
        example_summary_clean = str(annotated_summary['summary']).replace('\n', ' ').strip()
        if example_summary_clean == current_summary_clean:
            continue
        prompt_examples += f"\tSummary ({current_example_num}): \"{example_summary_clean}\"\n\n"
        current_example_num += 1
        for annotator_index, annotator_annotations in enumerate(annotated_summary['annotations']):
            prompt_examples += f"\t\tAnnotator ({chr(65 + annotator_index)}) Annotations:\n\n"
            if not annotator_annotations:
                prompt_examples += "\t\t\tNo Hallucinated Spans Identified by Annotator\n\n"
            else:
                for annotation in annotator_annotations:
                        if "source_span" in annotation and annotation["source_span"]:
                            reference_span = annotation["source_span"].replace("\n", " ").strip()
                            prompt_examples += f"\t\t\tReference Span from Source Article: \"{reference_span}\"\n"
                        if "summary_span" in annotation and annotation["summary_span"]:
                            summary_span = annotation["summary_span"].replace("\n", " ").strip()
                            prompt_examples += f"\t\t\tHallucinated Span from Summary: \"{summary_span}\"\n"
                        if annotation.get("label"):
                            labels = [lbl.split(".")[0] for lbl in annotation["label"]]
                            if "Unwanted" in labels:
                                label = "Unwanted"
                            elif "Benign" in labels:
                                label = "Benign"
                            else:
                                label = labels[0]
                            prompt_examples += f"\t\t\tLabel: \"{label}\"\n"
                        if annotation.get("note"):
                            note = annotation["note"].replace("\n", " ").strip()
                            prompt_examples += f"\t\t\tAnnotator Note: \"{note}\"\n"

                        prompt_examples += "\n"

    return prompt_examples.rstrip()

def evaluate_summaries(summaries, eval_data, output_dir, client, model_identifier,  max_retries=5):
    """
    Run hallucination evaluation for a single model's summaries using detailed annotated examples.
    The evaluation prompt is sent to a fixed evaluation model. Prints statistics after processing.
    """
    evaluation_model = "o3-mini"  # Fixed evaluation model
    
    # (Aggregated evaluation data mapping block removed; using eval_data directly)

    os.makedirs(output_dir, exist_ok=True)
    organization, model_name = model_identifier.split('/', 1)
    output_filename = os.path.join(output_dir, f"{model_name}_hallucination_eval.txt")

    total_samples = 0
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for source_id, (csv_source, summary) in tqdm(summaries.items(), desc="Evaluating samples"):
            if source_id not in eval_data:
                logging.warning(f"Source from source_id {source_id} not found in evaluation data. Skipping.")
                continue
            total_samples += 1

            annotated_summaries = eval_data[source_id]['annotated_summaries']

            prompt_examples = build_prompt_examples(annotated_summaries, summary)

            formatted_prompt = TEMPLATE.format(
                eval_data[source_id]['source'],
                prompt_examples,
                str(summary).replace('\n', ' ').strip()
            )
            
            request_args = {
                'model': evaluation_model,
                'messages': [{'role': 'user', 'content': formatted_prompt}],
                'reasoning_effort': 'high'
            }

            retries = 0
            while retries < max_retries:
                try:
                    response = client.chat.completions.create(**request_args)
                    break
                except Exception as e:
                    retries += 1
                    logging.error(f"Request failed for source_id {source_id} (attempt {retries}/{max_retries}): {e}")
                    time.sleep(2)
            else:
                logging.error(f"Exceeded max retries for source_id {source_id}. Skipping...")
                continue

            generated_text = response.choices[0].message.content

            response_dict = {
                'source_id': source_id,
                'summary': summary,
                'prompt': formatted_prompt, 
                'evaluation_response': generated_text
            }
            outfile.write(json.dumps(response_dict, ensure_ascii=False) + '\n')

    logging.info(f"Output saved to {output_filename}")

    # Compute and print hallucination and answer rates, and count invalid responses
    total, inconsistent, invalid_count = 0, 0, 0
    with open(output_filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                logging.error(f"Skipping invalid JSON line in {output_filename}")
                continue
            evaluation_response = result.get('evaluation_response', '')
            summary = result.get('summary', '')
            classification = extract_final_classification(evaluation_response)
            if classification is None:
                logging.warning(f"Could not extract classification for sample {result.get('source_id')} in {output_filename}")
                continue
            total += 1
            if classification == 'Inconsistent':
                inconsistent += 1
            if classification == 'Invalid':
                invalid_count += 1

    answer_rate = (total - invalid_count) / total if total > 0 else 0
    hallucination_rate = (inconsistent / total) if total > 0 else 0
    stats_str = (
        f"Total Evaluated Samples: {total}\n"
        f"Inconsistent Samples: {inconsistent}\n"
        f"Invalid Samples: {invalid_count}\n"
        f"Hallucination Rate: {hallucination_rate:.2%}\n"
        f"Answer Rate: {answer_rate:.2%}\n"
        + ("-" * 40) + "\n"
    )
    logging.info(stats_str)


def main():
    parser = argparse.ArgumentParser(description='Evaluate summaries for hallucinations for a single model.')
    parser.add_argument('--base_dir', type=str, default='generated_summaries', help='Directory containing generated summaries.')
    parser.add_argument('--jsonl_file', type=str, default='faithbench_summaries_by_source.jsonl', help='Path to the evaluation JSONL file.')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Directory to save evaluation results.')
    parser.add_argument('--model', type=str, required=True, help='Model identifier in the format organization/model.')
    args = parser.parse_args()

    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logging.error('OPENAI_API_KEY environment variable not set. Exiting.')
        return

    client = OpenAI(api_key=api_key)

    summaries = load_summary(args.base_dir, args.model)
    if not summaries:
        logging.error('No summaries loaded. Exiting.')
        return
    eval_data = load_eval_data(args.jsonl_file)
    if not eval_data:
        logging.error('No evaluation data loaded. Exiting.')
        return
    evaluate_summaries(summaries, eval_data, args.output_dir, client, args.model)


if __name__ == '__main__':
    main()