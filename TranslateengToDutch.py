# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import pipeline

# # Load the tokenizer and model from the local directory
# model_dir = "D:\FlareGPT\EnglishToDutchModel"
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# # Create a translation pipeline
# translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer)

# # Translate text
# input_text = "Hello, how are you?"
# translated_text = translation_pipeline(input_text, target_language="nl")

# print(translated_text)
from defaultValues import ED_TRANSLATOR_DIRECTORY

model_name = ED_TRANSLATOR_DIRECTORY

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
device_num = 0 if torch.cuda.is_available() else -1
device = "cpu" if device_num < 0 else f"cuda:{device_num}"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True).to(
    device
)

def my_ED_translator(english_text):
    params = {"max_length": 370, "num_beams": 4, "early_stopping": True}
    translator = pipeline("translation", tokenizer=tokenizer, model=model, device=device_num)
    # print(translator("Young Wehling was hunched in his chair, his head in his hand. He was so rumpled, so still and colorless as to be virtually invisible.",
    #          **params)[0]['translation_text'])   
    return(translator(english_text,
            **params)[0]['translation_text'])    

