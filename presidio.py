from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import json
from pprint import pprint

# text_to_anonymize = "His name is Mr. Jones and his phone number is 212-555-5555"
text_to_anonymize = "Taojie was an ordinary young man from the small fishing village of Yuanshui. \
                    Every morning, he would rise before dawn to help his father cast nets into the endless blue sea. \
                    By day’s end, he’d return with sunburnt cheeks and a small basket of fish, a routine he followed diligently. \
                        But Taojie heart yearned for something more—he longed for adventure, a life beyond the predictable rhythms of the tides."

analyzer = AnalyzerEngine()
# analyzer_results = analyzer.analyze(text=text_to_anonymize, entities=["PHONE_NUMBER"], language='en')
analyzer_results = analyzer.analyze(text=text_to_anonymize, entities=None, language='en')

print(analyzer_results) 


anonymizer = AnonymizerEngine()

anonymized_results = anonymizer.anonymize(
    text=text_to_anonymize,
    analyzer_results=analyzer_results,    
    operators={"DEFAULT": OperatorConfig("replace", {"new_value": ""}), 
                        "PHONE_NUMBER": OperatorConfig("mask", {"type": "mask", "masking_char" : "*", "chars_to_mask" : 12, "from_end" : True}),
                        "TITLE": OperatorConfig("redact", {})}
)

print(f"text: {anonymized_results.text}")
print("detailed response:")

pprint(json.loads(anonymized_results.to_json()))
