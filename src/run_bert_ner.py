from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import random
import time

DATASET = ['musique_s', 'wikimqa_s', 'samsum']
CHOICE = 1

# load the dataset
dataset = 'inputs/' + DATASET[CHOICE] + '.json'
eval_dataset = load_dataset(dataset)

# prepare prompts
prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

# init bert ner model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# prepare result file
file_name = "result_" + DATASET[CHOICE] + time.strftime("%Y%m%d-%H%M%S") + ".txt"
result_file = "results/" + file_name

# choose a random examples
flag = False
for _  in range(1):
    # choose a random example
    index = random.randint(0, len(eval_dataset))
    ex = eval_dataset[index]
    with open(result_file, "a") as f:
        f.write(f"Example {index}:\n")
    
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
    
    # analyze the example and mask it 
    ner_results = nlp(doc_prompts)
    masked_text = doc_prompts
    for entity in sorted(ner_results, key=lambda x: x['start'], reverse=True):
        start = entity['start']
        end = entity['end']
        masked_text = masked_text[:start] + "[MASK]" + masked_text[end:]
        
    # print out ner result
    print("ner result: ", ner_results, end="\n\n")
    
    # write the result into result_file
    with open(result_file, "a") as f:
        f.write(f"Original text: {doc_prompts}\n\n")
        f.write(f"Masked text: {masked_text}\n")
        f.write("\n\n")
    
    