from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import HTMLResponse

from fastapi import FastAPI, UploadFile, Form, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse

import os
import subprocess

from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request

app = FastAPI()



@app.get("/")
def read_html():
    with open("index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/uploadfile")
async def upload_file(file: UploadFile):
    os.makedirs("DutchToEnglishFiles_cID", exist_ok=True)
    file_path = os.path.join("DutchToEnglishFiles_cID", file.filename)
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    return {"filename": file.filename}

# @app.post("/runtranslate/")
# async def run_translate():
#     try:
#         subprocess.Popen(["python", "translate.py"])
#         return {"message": "Translation started in the background."}
#     except Exception as e:
#         return {"error": str(e)}

# @app.post("/runembed/")
# async def run_embed():
#     try:
#         subprocess.Popen(["python", "EmbedVecStore.py"])
#         return {"message": "Embedding started in the background."}
#     except Exception as e:
#         return {"error": str(e)}

#### to get the data input directly ####
from defaultValues import SOURCE_DIRECTORY, TRANSLATE_DIRECTORY
from fastapi import Form ##"pip install python-multipart"

@app.post("/extra_text")
async def user_text_input(user_data: str= Form(...)):
    with open(f"{TRANSLATE_DIRECTORY}/user_dutch_text.txt", "w") as user_input_file:
        user_input_file.write(user_data)
    return {"messege": "File created for the user text"}


    
@app.post("/process")
async def run_embed():
    try:
        # Start the first process
        print("\nTranslation started")
        process1 = subprocess.Popen(["python", "translate.py"])
        # Wait for the first process to finish
        process1.wait()
        print("\ntranslation completed")

        # Start the second process after the first one finishes
        process2 = subprocess.Popen(["python", "EmbedVecStore.py"])
        # Wait for the second process to finish
        process2.wait()
        return {"message": "Document Processing Completed."}
    except Exception as e:
        return {"error": str(e)}
    




class ChatRequest(BaseModel):
    user_prompt: str


### 1. FOR CPU CHATTING ###

import os
import logging
import click
import torch
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from TranslatedutToEnglish import my_DE_translator
from TranslateengToDutch import my_ED_translator

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)

from load_LLMs import (
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from defaultValues import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_ID2,
    MODEL_ID3,
    MODEL_BASENAME,
    MODEL_BASENAME3,
    MAX_NEW_TOKENS,
    MODELS_PATH,
)

def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        # top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


if torch.backends.mps.is_available():
    device_type = "mps"
elif torch.cuda.is_available():
    device_type = "cuda"
else:
    device_type = "cpu"

SHOW_SOURCES = True
logging.info(f"Running on: {device_type}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")


embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})
# uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# load the vectorstore
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
)
retriever = db.as_retriever()


# template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
# just say that you don't know, don't try to make up an answer.

# {context}

# {history}
# Question: {question}
# Helpful Answer:"""

# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
# prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
# memory = ConversationBufferMemory(input_key="question", memory_key="history")


# load the llm pipeline
# choose any model you like ie: llama2, vicuna, llama2cpu by uncommenting the llm loading lines below as per your choice.... if you have gpu use "gptq" else use the "gguf" option.

# # 1. For Vicuna (gptq)
# llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

# # 2. For Llama2 GPU (gptq) 
# llm = load_model(device_type, model_id=MODEL_ID2, model_basename=MODEL_BASENAME, LOGGING=logging)

# 3. For Llama2 cpu+gpu (gguf)
llm = load_model(device_type, model_id=MODEL_ID3, model_basename=MODEL_BASENAME3, LOGGING=logging)

prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
    retriever=retriever,
    return_source_documents=True,  # verbose=True,
    callbacks=callback_manager,
    chain_type_kwargs={"prompt": prompt, "memory": memory},
)
# else:
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
#         retriever=retriever,
#         return_source_documents=True,  # verbose=True,
#         callbacks=callback_manager,
#         chain_type_kwargs={
#             "prompt": prompt,
#         },
#     )

def chat_function(user_prompt: str) -> str:
    ###for english only
    # global qa  # making sure that QA is globally accessible
    # if user_prompt:
    #     res = qa(user_prompt)
    #     answer, docs = res["result"], res["source_documents"]
    #     return f"{answer}"
    
    ###for dutch
    global qa  # making sure that QA is globally accessible
    if user_prompt:
        QueToEnglish = my_DE_translator(user_prompt) # translate question to English
        res = qa(QueToEnglish) # returns response object in english....
        answer, docs = res["result"], res["source_documents"] # taking answer and docs info (in english)
        # printing in terminal
        print("\n\n> Question:") 
        print(user_prompt)
        print("\n> Answer From CPU:")
        print(answer)
        print("\n")
        AnsToDutch = my_ED_translator(answer)
        print("Answer In Dutch:", AnsToDutch) # print in console
        return f"{AnsToDutch}" # returning the answer from this chat_function to the caller....


@app.post("/chat/")
async def chat_with_bot(chat_request: ChatRequest):
    user_prompt = chat_request.user_prompt
    bot_response = chat_function(user_prompt)
    return {"bot_response": bot_response}


