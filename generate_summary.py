import time
import os
import requests
import json
import re
import csv
import argparse
import datetime
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from openai import OpenAI
from google import genai
from google.genai import types, errors
import anthropic
import glob

INPUT_FILE = "benchmarkz_summaries_by_source.jsonl"

system_prompt = (
    "You must respond based on the information provided in the passage. Do not incorporate any external knowledge or details beyond what is given."
)

base_user_prompt = (
    "Provide a concise summary of the following passage, covering the core pieces of information described:"
)

def load_model(model_id):
    """
    Load or initialize the model/client once, and return a dictionary
    containing everything needed for generation.
    """
    model_data = {"model_id": model_id, "client": None, "mode": None}

    if "openai" in model_id.lower():
        print(f"Loading OpenAI client for '{model_id}'...")
        model_data["client"] = OpenAI()
        model_data["mode"] = "openai"

    elif "google" in model_id.lower():
        print(f"Loading Google GenAI client for '{model_id}'...")
        model_data["mode"] = "google"

    elif "anthropic" in model_id.lower():
        print(f"Loading Anthropic client for '{model_id}'...")
        model_data["client"] = anthropic.Anthropic()
        model_data["mode"] = "anthropic"

    elif "deepseek" in model_id.lower():
        print(f"Loading DeepSeek client for '{model_id}'...")
        deepseek_client = OpenAI(api_key=os.getenv("DeepSeek_API_KEY"), base_url="https://api.deepseek.com")
        model_data["client"] = deepseek_client
        model_data["mode"] = "deepseek"

    elif "microsoft" in model_id.lower():
        print(f"Loading HuggingFace pipeline for Microsoft model '{model_id}' locally. This may be large...")
        pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=AutoTokenizer.from_pretrained(model_id),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        model_data["client"] = pipe
        model_data["mode"] = "microsoft"

    else:
        print(f"Will use OpenRouter for '{model_id}'...")
        model_data["mode"] = "openrouter"

    return model_data

def generate_summary(model_data, system_prompt, user_prompt, max_retries=100):
    """
    Uses the already-loaded model/client in model_data to generate a summary.
    If a 429 (RESOURCE_EXHAUSTED) error occurs, we wait and retry up to max_retries times.
    """
    attempt = 0
    model_id = model_data["model_id"]
    mode = model_data["mode"]
    client = model_data["client"]

    while attempt < max_retries:
        try:
            # -------------------------
            # 1) OPENAI
            # -------------------------
            if mode == "openai":
                print(f'Requesting {model_id} via OpenAI API...')
                if "gpt-3.5" in model_id.lower():
                    response = client.chat.completions.create(
                        model=model_id.replace('openai/',''),
                        messages=[
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0,
                        max_completion_tokens=4096,
                    )
                elif "gpt" in model_id.lower():
                    response = client.chat.completions.create(
                        model=model_id.replace('openai/',''),
                        messages=[
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0,
                        max_completion_tokens=8192,
                    )
                elif "o1" in model_id.lower():
                    response = client.chat.completions.create(
                        model=model_id.replace('openai/',''),
                        messages=[
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_completion_tokens=8192,
                    )
                elif "o3" in model_id.lower():
                    match = re.search(r'(low|medium|high)$', model_id)
                    if match:
                        think_mode = match.group(1)
                    else:
                        think_mode = "medium"
                    response = client.chat.completions.create(
                        model="o3-mini",
                        reasoning_effort=think_mode,
                        messages=[
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_completion_tokens=8192,
                    )
                else:
                    response = client.chat.completions.create(
                        model=model_id.replace('openai/',''),
                        messages=[
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0,
                        max_completion_tokens=8192,
                    )
                return response.choices[0].message.content

            # -------------------------
            # 2) GOOGLE
            # -------------------------
            elif mode == "google":
                print(f'Requesting {model_id} via Google API...')
                if "gemini-2.0" in model_id.lower():
                    if 'flash-thinking-exp' in model_id.lower():
                        prompt = system_prompt + ' ' + user_prompt
                        genai_client = genai.Client(api_key=os.getenv('GOOGLE_AI_API_KEY'), http_options={'api_version':'v1alpha'})
                        config = {
                            'thinking_config': {'include_thoughts': False}, 
                            "temperature": 0,
                        }
                        response = genai_client.models.generate_content(
                            model=model_id.lower().split('google/')[-1],
                            contents=prompt,
                            config=config,
                        )
                        if (response.candidates and
                            response.candidates[0].content and
                            response.candidates[0].content.parts and
                            len(response.candidates[0].content.parts) > 0):
                            return response.candidates[0].content.parts[0].text
                        else:
                            print("Warning: No valid text returned by flash-thinking-exp.")
                            return "No content returned by flash-thinking-exp model."
                    else:
                        genai_client = genai.Client(api_key=os.getenv('GOOGLE_AI_API_KEY'))
                        response = genai_client.models.generate_content(
                            model=model_id.lower().split('google/')[-1],
                            contents=user_prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=system_prompt,
                                max_output_tokens=8192,
                                temperature=0
                            )
                        )
                        return response.text
                else:
                    genai_client = genai.Client(api_key=os.getenv('GOOGLE_AI_API_KEY'))
                    response = genai_client.models.generate_content(
                        model=model_id.lower().split('google/')[-1],
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            max_output_tokens=8192,
                            temperature=0
                        )
                    )
                    return response.text

            # -------------------------
            # 3) ANTHROPIC
            # -------------------------
            elif mode == "anthropic":
                print(f'Requesting {model_id} via Anthropic API...')
                if "think" in model_id.lower():
                    message = client.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        max_tokens=8192,
                        temperature=0,
                        system=system_prompt,
                        thinking={
                            "type": "enabled",
                            "budget_tokens": 2048
                        },
                        messages=[{
                            "role": "user",
                            "content": user_prompt,
                        }]
                    )
                    return message.content[1].text
                else:
                    message = client.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        max_tokens=8192,
                        temperature=0,
                        system=system_prompt,
                        messages=[{
                            "role": "user",
                            "content": user_prompt,
                        }]
                    )
                    return message.content[0].text

            # -------------------------
            # 4) DEEPSEEK
            # -------------------------
            elif mode == "deepseek":
                print(f'Requesting {model_id} via DeepSeek API...')
                if 'v3' in model_id.lower():
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=8192,
                        temperature=0,
                        stream=False
                    )
                elif 'r1' in model_id.lower():
                    response = client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=8192,
                    )
                else:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=8192,
                        temperature=0
                    )
                return response.choices[0].message.content

            # -------------------------
            # 5) MICROSOFT/HF Pipeline
            # -------------------------
            elif mode == "microsoft":
                print(f"Generating with local pipeline for {model_id}...")
                pipe = client  
                messages = [
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}
                ]
                outputs = pipe(
                    messages,
                    max_new_tokens=8192,
                    do_sample=False
                )
                try:
                    return outputs[0]["generated_text"][-1]['content']
                except (KeyError, TypeError, IndexError):
                    return str(outputs[0].get("generated_text", "No output?"))
            # -------------------------
            # 6) OPENROUTER (fallback)
            # -------------------------
            else:
                print(f'Requesting {model_id} via OpenRouter API...')
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                        "Content-Type": "application/json"
                    },
                    data=json.dumps({
                        "model": model_id,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0,
                        "max_tokens": 8192,
                    })
                )
                return response.json()['choices'][0]['message']['content']

        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                attempt += 1
                wait_time = 60
                if attempt < max_retries:
                    print(f"Hit 429 RESOURCE_EXHAUSTED; waiting {wait_time}s before retry #{attempt}...")
                    time.sleep(wait_time)
                else:
                    print("Max retries exceeded. Returning error message.")
                    return f"ERROR: Quota exhausted after {max_retries} attempts. Could not complete request."
            else:
                raise

        except requests.RequestException as req_error:
            if "429" in str(req_error):
                attempt += 1
                wait_time = 60
                if attempt < max_retries:
                    print(f"Hit 429 rate-limit from openrouter; waiting {wait_time}s before retry #{attempt}...")
                    time.sleep(wait_time)
                else:
                    print("Max retries for openrouter exceeded.")
                    return f"ERROR: Quota exhausted after {max_retries} attempts (openrouter)."
            else:
                raise

    return "ERROR: Unknown condition ended generation loop."

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a summary from an LLM model for given passages."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/o3-mini",
        help="Model ID, e.g., openai/o3-mini, google/gemini-2.0, openai/o3-mini-high, etc."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="false",
        help="Whether to resume from the last generated CSV (true/false)."
    )
    return parser.parse_args()

def find_last_csv(out_dir):
    # Look for CSV files matching summaries_*.csv in the given directory
    csv_files = glob.glob(os.path.join(out_dir, "summaries_*.csv"))
    if not csv_files:
        return None
    csv_files.sort()
    return csv_files[-1]

def count_skip_lines(resume_source_id):
    """Count how many lines to skip until we reach the resume_source_id."""
    skip_count = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            skip_count += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if sample.get("source_id") == resume_source_id:
                break
    return skip_count

def main():
    args = parse_args()
    model_id = args.model
    resume_run = args.resume.lower() == "true"

    parts = model_id.split("/")
    if len(parts) == 2:
        organization, model_name = parts
    else:
        organization = parts[0]
        model_name = parts[0]

    out_dir = os.path.join("generated_summaries", organization, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # Determine resume position if needed
    resume_source_id = None
    last_source = None
    csv_path = None
    if resume_run:
        last_csv = find_last_csv(out_dir)
        if last_csv:
            print(f"Resuming from last CSV file: {last_csv}")
            csv_path = last_csv  # Append to the existing file
            with open(last_csv, "r", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    resume_source_id = last_row[0]  # source_id is in first column
                    last_source = last_row[1]       # source is in second column
                    print(f"Last processed source_id: {resume_source_id}")
                else:
                    print("CSV file is empty. Starting from the beginning.")
        else:
            print("No previous CSV found. Starting a new CSV.")
    if not csv_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = f"summaries_{timestamp}.csv"
        csv_path = os.path.join(out_dir, csv_name)

    # Count total lines in the input file.
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        total_samples = sum(1 for _ in f)

    # If resuming and a resume_source_id is found, count how many lines to skip.
    if resume_run and resume_source_id is not None:
        skip_lines = count_skip_lines(resume_source_id)
    else:
        skip_lines = 0

    remaining_total = total_samples - skip_lines

    # 1) LOAD the model/client ONCE
    print(f"\n=== Loading model/client for '{model_id}' ===")
    model_data = load_model(model_id)

    # Open the input file; open the CSV in append mode if resuming
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(csv_path, "a", encoding="utf-8", newline="") as csvfile:

        writer = csv.writer(csvfile)
        # If the CSV is new, write header
        if os.stat(csv_path).st_size == 0:
            writer.writerow(["source_id", "source", "summary"])

        # Skip already processed lines if resuming
        if resume_run and resume_source_id is not None:
            for _ in range(skip_lines):
                next(fin, None)

        resume_found = True  # We already skipped processed lines
        current_last_source = last_source  # for duplicate filtering

        # Use tqdm with the adjusted remaining_total
        for line in tqdm(
            fin,
            total=remaining_total,
            desc="Processing samples",
            unit="line",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} left: {remaining}]"
        ):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line}\n{e}")
                continue

            source_id = sample.get("source_id")
            source = sample.get("source")

            if not source:
                print(f"Skipping sample {source_id} due to missing source.")
                continue

            # Duplicate-check: skip if this sample's source is the same as the last processed
            if current_last_source is not None and source == current_last_source:
                print(f"Skipping sample {source_id} (same source as last).")
                continue
            else:
                current_last_source = source

            current_prompt = f"{base_user_prompt}\nPassage:\n{source}"
            summary = generate_summary(model_data, system_prompt, current_prompt)

            print("\n" + "="*60)
            print(f"SOURCE ID: {source_id}")
            print("="*60)
            print("SOURCE TEXT:")
            print(source)
            print("-"*60)
            print("GENERATED SUMMARY:")
            print(summary)
            print("="*60 + "\n")

            writer.writerow([source_id, source, summary])

    print(f"\nAll summaries appended to: {csv_path}")

if __name__ == "__main__":
    main()