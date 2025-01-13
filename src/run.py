from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import random
import time

DATASET = ['musique_s', 'wikimqa_s', 'samsum']
CHOICE = 2

# load the dataset
dataset = 'inputs/' + DATASET[CHOICE] + '.json'
eval_dataset = load_dataset(dataset)

# prepare prompts
prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

# init presidio analyzer and anonymizer
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# prepare result file
file_name = "result_" + DATASET[CHOICE] + time.strftime("%Y%m%d-%H%M%S") + ".txt"
result_file = "results/" + file_name

# choose a random examples
flag = False
for _  in range(1):
    # choose a random example
    index = random.randint(0, len(eval_dataset))
    ex = eval_dataset[index]
    
    # parse the example
    answers = ex["answers"]
    doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
    doc_prompts = ".".join([doc.strip() for doc in doc_prompts])
    
    # print(f"type of doc_prompts: {type(doc_prompts)}")
    # print(f"length of doc_prompts: {len(doc_prompts)}")
    
    # print the example out
    # if not flag:
    #     print(f"the prompt: {doc_prompts}")
    #     flag = True
    
    # analyze the example
    analyzer_results = analyzer.analyze(text=doc_prompts, entities=None, language='en')
    anonymized_results = anonymizer.anonymize(
        text=doc_prompts,
        analyzer_results=analyzer_results)
    
    # write the result into result_file
    # print(f"original text: {doc_prompts}")
    # print(f"analyzer results: {anonymized_results.text}")
    # print("\n\n")
    with open(result_file, "a") as f:
        f.write(f"Original text: {doc_prompts}\n")
        f.write(f"Anonymized text: {anonymized_results.text}\n")
        f.write("\n\n")
    
    