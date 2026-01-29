## LLM prompt for refining abstracts
abstract_refine_system_prompt ='''
Do not use any tools.
You are a precise academic summarizer. 
Output the core contribution and methodology in one concise sentence. 
Do not use conversational fillers like 'The paper discusses...' or 'This study presents...'.
Start your response immediately with the requested information.\n\n
Title: {title}\n
Abstract: {full_summary}'''

CANNOT_ANSWER_PHRASE = "I cannot answer"

answer_prompt = (
    "Answer the question below with the context.\n\n"
    "Context:\n\n{context}\n\n---\n\n"
    "Question: {question}\n\n"
    "Write an answer based on the context. "
    "If the context provides insufficient information reply "
    f'"{CANNOT_ANSWER_PHRASE}." '
    "For each part of your answer, indicate which sources most support "
    "it via citation keys at the end of sentences, like \n\n"
    "{example_citation}\n\n. "
    "Only cite from the context above and only use the citation keys from the context.\n\n"
    # f"\n\n{CITATION_KEY_CONSTRAINTS}\n\n"
    "Do not concatenate citation keys, just use them as is. "
    "Write in the style of a scientific article, with concise sentences and "
    "coherent paragraphs. This answer will be used directly, "
    "so do not add any extraneous information.\n\n"
    "{prior_answer_prompt}"
)

