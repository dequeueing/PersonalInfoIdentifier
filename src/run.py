from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

import random
import time

DATASET = ['musique_s', 'wikimqa_s', 'samsum']

# init presidio analyzer and anonymizer
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# prepare prompts
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

# init bert ner model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

for dataset in DATASET:
    # prepare file name
    file_name_presidio = "presidio_" + dataset + time.strftime("%Y%m%d-%H%M%S") + ".txt"
    file_name_bertner = "bertner_" + dataset + time.strftime("%Y%m%d-%H%M%S") + ".txt"
    result_file_presidio = "results/presidio/" + file_name_presidio
    result_file_bertner = "results/bert-ner/" + file_name_bertner
    
    # choose a random example from the dataset
    dataset = 'inputs/' + dataset + '.json'
    eval_dataset = load_dataset(dataset)
    index = random.randint(0, len(eval_dataset))
    ex = eval_dataset[index]
    with open(result_file_presidio, "a") as f:
        f.write(f"Dataset: {dataset} {index}th example:\n")
    with open(result_file_bertner, "a") as f:
        f.write(f"Dataset: {dataset} {index}th example:\n")

    # get a text chunk from the example
    doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
    doc_prompts = doc_prompts[0].strip()   # str

    # get ner result
    bertner_result = nlp(doc_prompts)
    analyzer_results = analyzer.analyze(text=doc_prompts, entities=None, language='en')
    anonymized_results = anonymizer.anonymize(
        text=doc_prompts,
        analyzer_results=analyzer_results)
    
    # for ner-bert, replace with [MASK]
    masked_text = doc_prompts
    for entity in sorted(bertner_result, key=lambda x: x['start'], reverse=True):
        start = entity['start']
        end = entity['end']
        masked_text = masked_text[:start] + "[MASK]" + masked_text[end:]
        
    # write the result into corresponding result_file
    with open(result_file_presidio, "a") as f:
        f.write(f"Original Text: {doc_prompts}\n")
        f.write(f"Anonymized Text: {anonymized_results.text}\n\n")
    with open(result_file_bertner, "a") as f:
        f.write(f"Original Text: {doc_prompts}\n")
        f.write(f"Anonymized Text: {masked_text}\n\n")        
    
    