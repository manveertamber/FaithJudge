class PromptTemplates:


    LUO_COT_PROMPT = '''Decide whether the following summary is consistent with the corresponding article. A summary is consistent if all of its information is fully supported by the article. If any part of the summary is unsupported, contradictory, or hallucinated, classify it as inconsistent.

---

The article and summary are given below.
Source Article:
"{}"

Summary:
"{}"

---

Explain your reasoning step by step, then provide a clear answer at the end of your response in this format: "Final classification: [Consistent / Inconsistent]". Do not include any other text after the final classification.'''


    RAGAS_PROMPT = '''Consider the given context and following statements, then determine whether the statements are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Consistent or Inconsistent). At the end, provide a final verdict. If all statements are consistent, the final verdict should be Consistent. If any statement is inconsistent, the final verdict should be Inconsistent.

---
Context: "{}"

{}

---

Provide a clear answer at the end of your response in this format: \"Final classification: [Consistent / Inconsistent]\". Do not include any other text after the final classification.
'''


    CLAIM_EXTRACTION = '''Segment the following sentence completely into individual independent statements. Maintain exactly what is asserted, without adding or removing meaning, and avoid making any statements more general than originally expressed. Respond only with the statements. First, some examples are provided for guidance:
Sentence: Other title changes included Lord Steven Regal and The Nasty Boys winning the World Television Championship and the World Tag Team Championship respectively.
Statements:
- Other title changes include Lord Steven Regal winning the World Television Championship.
- Other title changes include The Nasty Boys winning the World Tag Team Championship.

Sentence: The parkway was opened in 2001 after just under a year of construction and almost two decades of community requests.
Statements:
- The parkway was opened in 2001.
- The parkway was opened after just under a year of construction.
- The parkway was opened after almost two decades of community requests.

Sentence: Touring began in Europe in April-June with guitarist Paul Gilbert as the opening act, followed by Australia and New Zealand in July, Mexico and South America in late July-August, and concluding in North America in October-November.
Statements:
- Touring began in Europe in April-June.
- The opening act of the tour was guitarist Paul Gilbert.
- The tour was in Australia and New Zealand in July.
- The tour was in Mexico and South America in late July-August.
- The tour was concluded in North America in October-November.

Sentence: In March 2018, the company partnered with Amazon Web Services (AWS) to offer AI-enabled conversational solutions to customers in India.
Statements:
- The company partnered with Amazon Web Services (AWS) in March 2018.
- The company partnered with AWS to offer AI-enabled conversational solutions to customers in India.

Sentence: The most significant of these is in Germany, which now has a Yazidi community of more than 200,000 living primarily in Hannover, Bielefeld, Celle, Bremen, Bad Oeynhausen, Pforzheim and Oldenburg.
Statements:
- The most significant of these is in Germany.
- Germany now has a Yazidi community of more than 200,000.
- The Yazidi community in Germany lives primarily in Hannover, Bielefeld, Celle, Bremen, Bad Oeynhausen, Pforzheim and Oldenburg.

Sentence: A previous six-time winner of the Nations’ Cup, Sebastian Vettel became Champion of Champions for the first time, defeating Tom Kristensen, who made the final for the fourth time, 2-0.
Statements:
- Sebastian Vettel is a previous six-time winner of the Nations’ Cup.
- Sebastian Vettel became Champion of Champions for the first time, defeating Tom Kristensen, 2-0.
- Tom Kristensen made the final for the fourth time.

Sentence: Cats can fly short distances using their retractable wings.
Statements:
- Cats can fly short distances.
- Cats have retractable wings.

Sentence: {}
Statements:'''



    FAITHBENCH_THREEWAY_CLASSIFICATION_WITH_ARTICLE_EXAMPLES = '''Your task is to evaluate a summary and determine whether it contains hallucinations, such as claims, details, or contradictions that are not supported by the source article.

You will be provided with the source article and annotated examples of summaries of the article. These examples contain hallucinations independently identified by human annotators, categorized as Benign, Unwanted, or Questionable, along with brief explanations. Use these annotated examples to guide your analysis.

Hallucination Categories:
• Benign: Information not present in the article but reasonable, supported by world knowledge, common sense, or logical reasoning, thus acceptable to readers.
• Unwanted: Problematic hallucinations, including contradictions, misrepresentations, or unsupported details.
• Questionable: Possible hallucinations open to interpretation, where annotators might reasonably disagree.

Your task is to provide a final classification of the summary as follows:
• Consistent: Summary contains no hallucinations.
• Benign: Summary contains only Benign hallucinations.
• Unwanted: Summary contains any Unwanted hallucinations.

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

Provide your reasoning first, then clearly state your final classification at the end in this format:
"Final classification: [Consistent / Benign / Unwanted]"
If the provided text is unrelated to the article and does not summarize it, state clearly: "Final classification: Invalid".
Do not include any additional text after the final classification.'''



    FAITHBENCH_BINARY_CLASSIFICATION_WITH_ARTICLE_EXAMPLES = '''Your task is to evaluate a summary and determine whether it contains hallucinations, such as claims, details, or contradictions that are not supported by the source article.

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


    FACTS_GROUNDING_JSON = '''You are a helpful and harmless AI assistant. You will be provided with a textual context and a model-generated response.
Your task is to analyze the response sentence by sentence and classify each sentence according to its relationship with the provided context.

**Instructions:**

1. **Decompose the response into individual sentences.**
2. **For each sentence, assign one of the following labels:**
    * **"supported"**: The sentence is entailed by the given context. Provide a supporting excerpt from the context. The supporting excerpt must *fully* entail the sentence. If you need to cite multiple supporting excerpts, simply concatenate them.
    * **"unsupported"**: The sentence is not entailed by the given context. No excerpt is needed for this label.
    * **"contradictory"**: The sentence is falsified by the given context. Provide a contradicting excerpt from the context.
    * **"no_rad"**: The sentence does not require factual attribution (e.g., opinions, greetings, questions, disclaimers). No excerpt is needed for this label.
3. **For each label, provide a short rationale explaining your decision.** The rationale should be separate from the excerpt.
4. **Be very strict with your "supported" and "contradictory" decisions.** Unless you can find straightforward, indisputable evidence excerpts *in the context* that a sentence is "supported" or "contradictory", consider it "unsupported". You should not employ world knowledge unless it is truly trivial.

**Input Format:**

The input will consist of two parts, clearly separated:
    * **Context:** The textual context used to generate the response.
    * **Response:** The model-generated response to be analyzed.

**Output Format:**

For each sentence in the response, output a JSON object with the following fields:
    * "sentence": The sentence being analyzed.
    * "label": One of "supported", "unsupported", "contradictory", or "no_rad".
    * "rationale": A brief explanation for the assigned label.
    * "excerpt": A relevant excerpt from the context. Only required for "supported" and "contradictory" labels.

Output each JSON object on a new line.

**Example:**
    **Input:**
        Context: Apples are red fruits. Bananas are yellow fruits.
        Response: Apples are red. Bananas are green. Bananas are cheaper than apples. Enjoy your fruit!

    **Output:**
        {{"sentence": "Apples are red.", "label": "supported", "rationale": "The context explicitly states that apples are red.", "excerpt": "Apples are red fruits."}}
        {{"sentence": "Bananas are green.", "label": "contradictory", "rationale": "The context states that bananas are yellow, not green.", "excerpt": "Bananas are yellow fruits."}}
        {{"sentence": "Bananas are cheaper than apples.", "label": "unsupported", "rationale": "The context does not mention the price of bananas or apples.", "excerpt": null}}
        {{"sentence": "Enjoy your fruit!", "label": "no_rad", "rationale": "This is a general expression and does not require factual attribution.", "excerpt": null}}

**Now, please analyze the following context and response:**

**User Query:**
    Summarize the following context.

**Context:**
    {}

**Response:**
    {}

'''
