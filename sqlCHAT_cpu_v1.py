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
from load_LLMs import (
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

            db=SQLDatabase.from_uri("sqlite:///FlareGPT/My_database.db")
            db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
            while True:
                que=input("1. Enter the query in NL or exit to stop:")
                if que=="exit":
                    break
                # response = chain.invoke({"question": "How many Artist are there, here the table name is 'Artist'"})
                try:
                  # db_chain.run(que)
                  response=db_chain.run(que)
                  print("/nBOT:",response)
                except Exception as e:
                  print("There should be the required Table in DB. The error is :",e)
            
            

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
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)

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

        elif Method_choice == "5":
            ## Extending THE SQL TOOLKIT
            # from langchain.agents import create_sql_agent
            # from langchain.agents.agent_toolkits import SQLDatabaseToolkit
            # from langchain.agents.agent_types import AgentType
            # from langchain.sql_database import SQLDatabase

            few_shots = {
                            "List all artists.": "SELECT * FROM artists;",
                            "Find all albums for the artist 'AC/DC'.": "SELECT * FROM albums WHERE ArtistId = (SELECT ArtistId FROM artists WHERE Name = 'AC/DC');",
                            "List all tracks in the 'Rock' genre.": "SELECT * FROM tracks WHERE GenreId = (SELECT GenreId FROM genres WHERE Name = 'Rock');",
                            "Find the total duration of all tracks.": "SELECT SUM(Milliseconds) FROM tracks;",
                            "List all customers from Canada.": "SELECT * FROM customers WHERE Country = 'Canada';",
                            "How many tracks are there in the album with ID 5?": "SELECT COUNT(*) FROM tracks WHERE AlbumId = 5;",
                            "Find the total number of invoices.": "SELECT COUNT(*) FROM invoices;",
                            "List all tracks that are longer than 5 minutes.": "SELECT * FROM tracks WHERE Milliseconds > 300000;",
                            "Who are the top 5 customers by total purchase?": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM invoices GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
                            "Which albums are from the year 2000?": "SELECT * FROM albums WHERE strftime('%Y', ReleaseDate) = '2000';",
                            "How many employees are there": 'SELECT COUNT(*) FROM "employee"',
                        }

            #   from langchain.embeddings.openai import OpenAIEmbeddings
            from langchain.schema import Document
            from langchain.vectorstores import FAISS

            #   embeddings = OpenAIEmbeddings()
            embeddings = embeddings
            
            few_shot_docs = [
                            Document(page_content=question, metadata={"sql_query": few_shots[question]})
                            for question in few_shots.keys()
                            ]
            
            vector_db = FAISS.from_documents(few_shot_docs, embeddings)
            retriever = vector_db.as_retriever()

            # Now we can create our own custom tool and append it as a new tool in the create_sql_agent function.
            from langchain.agents.agent_toolkits import create_retriever_tool

            tool_description = """
            This tool will help you understand similar examples to adapt them to the user question.
            Input to this tool should be the user question.
            """

            retriever_tool = create_retriever_tool(
                retriever, name="sql_get_similar_examples", description=tool_description
            )
            custom_tool_list = [retriever_tool]


            from langchain.agents import AgentType, create_sql_agent
            from langchain.agents.agent_toolkits import SQLDatabaseToolkit
            from langchain.chat_models import ChatOpenAI
            from langchain.utilities import SQLDatabase

            db = SQLDatabase.from_uri("sqlite:///FlareChat/My_database.db")
            #   llm = ChatOpenAI(model_name="gpt-4", temperature=0)
            llm = llm

            toolkit = SQLDatabaseToolkit(db=db, llm=llm)

            custom_suffix = """
            I should first get the similar examples I know.
            If the examples are enough to construct the query, I can build it.
            Otherwise, I can then look at the tables in the database to see what I can query.
            Then I should query the schema of the most relevant tables
            """

            agent_executor = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                #   agent_type=AgentType.OPENAI_FUNCTIONS,
                agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                extra_tools=custom_tool_list,
                suffix=custom_suffix,
            )

            # agent_executor = create_sql_agent(
            #       llm = llm,
            #       toolkit = toolkit,
            #       verbose = True,
            #       agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,)

            try:
                user_input=input("5. Please enter your query in NL:")
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

