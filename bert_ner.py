from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# 创建NER管道
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# 输入文本
example = "My name is Wolfgang and I live in Berlin"

# 获取NER结果
ner_results = nlp(example)

# 创建一个新的字符串，逐步替换敏感信息为 [MASK]
masked_text = example

# 按结果中的 start 和 end 替换敏感信息
for entity in sorted(ner_results, key=lambda x: x['start'], reverse=True):  # 倒序替换，防止索引错位
    start = entity['start']
    end = entity['end']
    masked_text = masked_text[:start] + "[MASK]" + masked_text[end:]

# 输出被 mask 的文本
print("Original Text:", example)
print("Masked Text:", masked_text)
