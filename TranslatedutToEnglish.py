# 1. Direct Translate for smaller but not for large.
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

from defaultValues import TRANSLATE_DIRECTORY, SOURCE_DIRECTORY, DE_TRANSLATOR_DIRECTORY
# 2. Translate for 370 tokens only (as in orignal DutchtoEnglish on yhavinga/ul2-base-en-nl).

Trans_DE_model_name = DE_TRANSLATOR_DIRECTORY

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import pipeline
import torch

device_num = 0 if torch.cuda.is_available() else -1
device = "cpu" if device_num < 0 else f"cuda:{device_num}"

tokenizer = AutoTokenizer.from_pretrained(Trans_DE_model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(Trans_DE_model_name, use_auth_token=True).to(
    device
)
def my_DE_translator(dutch_text):
    params = {"max_length": 4000, "num_beams": 4, "early_stopping": False}
    DE_translator = pipeline("translation", tokenizer=tokenizer, model=model, device=device_num)

    # print(translator("Mijn naam is Shishir",
    #         **params)[0]['translation_text'])  
    return(DE_translator(dutch_text,
            **params)[0]['translation_text'])


