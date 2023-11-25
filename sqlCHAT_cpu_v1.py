# first install langchain_experimental in the environment....
#pip install langchain_experimental
# for running on cpu install llama-cpp-python....pip install llama-cpp-python==0.1.83

import logging
import click
import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

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

# Added for sql
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
# from langchain.chains.sql_database.base import SQLDatabaseChain
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains import create_sql_query_chain


# # from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.vectorstores import Chroma

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

##for gpu
#from defaultValues import EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_ID2, MODEL_BASENAME

#for Cpu
from load_models_forCPU import (
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

#for cpu model....
from defaultValues import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID3,
    MODEL_BASENAME3,
    MAX_NEW_TOKENS,
    MODELS_PATH,
)


# from deep_translator import GoogleTranslator

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


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)

def main(device_type, show_sources):
    """
    This function implements the information retrieval task.


    1. Loads an embedding model, can be HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings
    2. Loads the existing vectorestore that was created by inget.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")

    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})


    #load cpu LLM
    llm = load_model(device_type, model_id=MODEL_ID3, model_basename=MODEL_BASENAME3)

    #### loading Chinook  ####
    # We need to first keep the Chinook.db in the same working folder ....Hence to create Chinook.db in the same directory as this notebook:
    # 1. Save the file from this link(https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql) to your sql quring code directory by naming it as Chinook_Sqlite.sql
    # 2. Then run these on terminal to make the Chinbook.db file ....
        # Run "sqlite3 Chinook.db" in terminal
        # Run ".read Chinook_Sqlite.sql"
    # 3. Test if it is working by writing this on the terminal's sql interface....
        # Test "SELECT * FROM Artist LIMIT 10;"
    # Now, Chinhook.db is in our directory which can be taken here by writing db=SQLDatabase.from_uri("sqlite:///Chinook.db") if the Chinook.db is in this folder....
    # db=SQLDatabase.from_uri("sqlite:///Chinook.db")
    
    from langchain.utilities import SQLDatabase
    from langchain.chains import create_sql_query_chain
    from langchain_experimental.sql import SQLDatabaseChain

    while True:
      
        Method_choice=input("Enter the Method you want to use.... 1. Using SQLDatabaseChain     2. Using create_sql_query_chain with same db   3. Using create_sql_query_chain with empty_db and then running on another db (actual Chinook) or STOP to stop this loop")

        if Method_choice=="STOP":
            break

        elif Method_choice == "1":

            # # # method 1: USING SQLDatabaseChain: (to direcly create the SQLDatabaseChain and run it (this direcly converts the text to sql and in the same time runs it to query the sql....))
            # # for this do following imports:
            # # from langchain.utilities import SQLDatabase
            # # from langchain_experimental.sql import SQLDatabaseChain

            db=SQLDatabase.from_uri("sqlite:///Chinook.db")
            db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
            try:
              db_chain.run("How many Artist in the Artist table?")
            except Exception as e:
              print("There should be Artist table")
            

        elif Method_choice == "2":
        # # # method 2: USING import create_sql_query_chain: ( to only produce the Sql first and then run that later....# )
        # from langchain.chains import create_sql_query_chain
            ## way1 : Using the same db to create the sql_query_chain as well as to run the response.... 
            # step 1: creating the sql 
            db=SQLDatabase.from_uri("sqlite:///Chinook.db")
            chain = create_sql_query_chain(llm, db)
            while True:
                que=input("Enter the query in NL or exit to stop:")
                if que=="exit":
                    break
                # response = chain.invoke({"question": "How many Artist are there, here the table name is 'Artist'"})
            
                response = chain.invoke({"question":que})
                print(response)

                # way1 : step 2: running the sql
                # db.run(response)
                try:
                  print("/n BOT:",db.run(response))
                except Exception as e:
                  print("The Table is not there")
            

        elif Method_choice == "3":

            # way2: Using empty_db to create the sql_query_chain and run the response with actuall db....(saves time in converting text to query....)
            #for this create a file with extension ".db" named Empty.db in this project folder itself....let Empty.db be db1 and the Chinook.db be db2....
            db1=SQLDatabase.from_uri("sqlite:///Empty.db")
            db=SQLDatabase.from_uri("sqlite:///Chinook.db")
            chain = create_sql_query_chain(llm, db1)
            while True:
                que=input("enter the query in NL or exit to stop:")
                if que=="exit":
                    break
                response = chain.invoke({"question":que})
                print(response)
                try:
                  print("/n BOT:",db.run(response))
                except Exception as e:
                  print("No table of that query.")

        elif Method_choice == "4":
            from langchain.agents import create_sql_agent
            from langchain.agents.agent_toolkits import SQLDatabaseToolkit
            from langchain.agents.agent_types import AgentType
            from langchain.sql_database import SQLDatabase

            db=SQLDatabase.from_uri("sqlite:///FlareGPT/Chinook.db")
            toolkit = SQLDatabaseToolkit(db=db, llm=llm(temperature=0))

            agent_executor = create_sql_agent(
                    llm = llm(temperature=0),
                    toolkit = toolkit,
                    verbose = True,
                    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,)

            # agent_executor.run("Describe the playlisttrack table")
            try:
                user_input=input("Please enter your query in NL:")
                agent_executor.run(user_input)
            except Exception as e:
                print("There is an error: ",e)

        else:
            print("Enter choice number: 1 or 2 or 3 or 4")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()

