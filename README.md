
# This is a Chat App that can answer user's query related to their uploaded documents' contents.

## The project is runable completely locally on users computer.... There are 3 different ways to interact with this chat app....
1. Web Based UI for answering Dutch documents from the brower....
2. English Document aswering in local terminal....
3. Dutch Document answering in local terminal....

# Please use the FlareGPT conda environment with python 3.10.0 for the Flare FastAPI and all these chatting code.... 

# In this project:
- GPT4ALL model replaced with models like Vicuna-7B or Llama2.
- Use of InstructorEmbeddings instead of LlamaEmbeddings from the original privateGPT.
- Both Embeddings and LLM now run on GPU as well as CPU as per   choice instead of only CPU.
- CPU support available as an option for users without a GPU.
- works w/o internet connection
- 100% safe
- ingest the document by running EmbedVecStore.py, and ask question using FlareChat.py
- Built using LangChain, llama2 or Vicuna 7B, and instructorEmbedding
([LangChain](https://github.com/hwchase17/langchain), [llama2](https://huggingface.co/TheBloke/Llama-2-7B-GPTQ) or [Vicuna-7B](https://huggingface.co/TheBloke/vicuna-7B-1.1-HF) (+ alot more!) and [InstructorEmbeddings](https://instructor-embedding.github.io/)). The models alternatives( for embedding and also for LLM) are in the defaultValues.py file, uncomment the one you want to use.



# Environment Setup:

Install conda

```shell
conda create -n FlareGPT python=3.10.0

conda activate FlareGPT

pip install -r requirements.txt

pip install -r otherrequirements.txt

#for cpu : pip install llama-cpp-python==0.1.83
```

# I. Run the project on web browser UI(FastAPI):
```shell
uvicorn FastAPI_FlareChat:app --reload
```
the server runs on "http://127.0.0.1:8000/" . Once the server is created you can try the demo by following instructions below.

## Try demo in the browswer at http://127.0.0.1:8000/:

### Step 1: Choose and Upload the document by clicking the upload button...

### Step 2: click the "Document Processing" button

### Step 3: Subsequently, after the document processing completes you can Start Chatting with your Dutch document by Clicking the Chat Button....


# II. Running the demo for English in the terminal ( Not Browser)....

### Step 1: Placing the documents in the folder

Put your docuemnts that can be in different formats (such as .txt, .pdf, or .csv files) into the SOURCE_DOCUMENTS directory

The current default file types are .txt, .pdf, .csv, and .xlsx, if you want to use any other file type, you will need to convert it to one of the default file types.


### Step2 : Ingesting your own dataset

Run the following command in the terminal to ingest all the data.

- python EmbedVecStore.py

Use the device type argument to specify a given device.
- python EmbedVecStore.py --device_type cpu

Use help for a full list of supported devices.
- python EmbedVecStore.py --help


It will create an index containing the local vectorstore. Will take time, depending on the size of your documents.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `index`.

Note: When you run this for the first time, it will download take time as it has to download the embedding model. In the subseqeunt runs, no data will leave your local enviroment and can be run without internet connection.

### Step 3: Ask question to chat in Terminal for English !!
In order to ask a question, run a command like:

```shell
python FlareChat.py
```

And wait for the script to require your input.

```shell
> Enter a query:
```

Hit enter. Wait while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again.

Note: When you run this for the first time, it will need internet connection to download the vicuna-7B model. After that you can turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.

## Trying the demo for Dutch in the terminal....

### Step 1: Placing the documents in the folder

Put your docuemnts that can be in different formats (such as .txt, .pdf, or .csv files) into the "DutchToEnglishFiles_cID" directory

The current default file types are .txt, .pdf, .csv, and .xlsx, if you want to use any other file type, you will need to convert it to one of the default file types.

### Step 2: Translating the documents

- python Translate.py


### Step2 : Ingesting your own dataset

Run the following command in the terminal to ingest all the data.

- python EmbedVecStore.py

Use the device type argument to specify a given device.
- python EmbedVecStore.py --device_type cpu

Use help for a full list of supported devices.
- python EmbedVecStore.py --help


It will create an index containing the local vectorstore. Will take time, depending on the size of your documents.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `index`.

Note: When you run this for the first time, it will download take time as it has to download the embedding model. In the subseqeunt runs, no data will leave your local enviroment and can be run without internet connection.

### Step 3: Ask question locally in Terminal for Dutch
In order to ask a question, run a command like:

```shell
python FlareChat_Dutch.py
```

And wait for the script to require your input.

```shell
> Enter a query:
```

Hit enter. Wait while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again.

Note: When you run this for the first time, it will need internet connection to download the vicuna-7B model. After that you can turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.

# Run it on CPU

By default, FlareGPT will use your GPU to run both the `EmbedVecStore.py` and `FLareChat.py` scripts. But if you do not have a GPU and want to run this on CPU, now you can do that (Warning: Its going to be slow!). You will need to use `--device_type cpu`flag with both scripts.

For Ingestion run the following:

```shell
python EmbedVecStore.py --device_type cpu
```

In order to ask a question, run a command like:

```shell
python FlareChat.py --device_type cpu
```
for dutch 
```shell
python FlareChat.py --device_type cpu
```

# Run quantized for M1/M2:

GGML quantized models for Apple Silicon (M1/M2) are supported through the llama-cpp library, [example](https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GGML). GPTQ quantized models that leverage auto-gptq will not work, [see here](https://github.com/PanQiWei/AutoGPTQ/issues/133#issuecomment-1575002893). GGML models will work for CPU or MPS.

## Troubleshooting

**Install MPS:**
1- Follow this [page](https://developer.apple.com/metal/pytorch/) to build up PyTorch with Metal Performance Shaders (MPS) support. PyTorch uses the new MPS backend for GPU training acceleration. It is good practice to verify mps support using a simple Python script as mentioned in the provided link.

2- By following the page, here is an example of what you may initiate in your terminal

```shell
xcode-select --install
conda install pytorch torchvision torchaudio -c pytorch-nightly
pip install chardet
pip install cchardet
pip uninstall charset_normalizer
pip install charset_normalizer
pip install pdfminer.six
pip install xformers
```

**Upgrade packages:**
Your langchain or llama-cpp version could be outdated. Upgrade your packages by running install again.

```shell
pip install -r requirements.txt
```

If you are still getting errors, try installing the latest llama-cpp-python with these flags, and [see thread](https://github.com/abetlen/llama-cpp-python/issues/317#issuecomment-1587962205).

```shell
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
```


# System Requirements

## Python Version

To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler

If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

### For Windows 10/11

To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
   - Universal Windows Platform development
   - C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the "gcc" component.

### NVIDIA Driver's Issues:

Follow this [page](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04) to install NVIDIA Drivers.


# Disclaimer

This is a test project to validate the feasibility of a fully local solution for question answering using LLMs and Vector embeddings. It is not production ready, and it is not meant to be used in production. Vicuna-7B is based on the Llama model so that has the original Llama license.

# Common Errors

 - [Torch not compatible with CUDA enabled](https://github.com/pytorch/pytorch/issues/30664)

   -  Get CUDA version
      ```shell
      nvcc --version
      ```
      ```shell
      nvidia-smi
      ```
   - Try installing PyTorch depending on your CUDA version
      ```shell
         conda install -c pytorch torchvision cudatoolkit=10.1 pytorch
      ```
   - If it doesn't work, try reinstalling
      ```shell
         pip uninstall torch
         pip cache purge
         pip install torch -f https://download.pytorch.org/whl/torch_stable.html
      ```

- [ERROR: pip's dependency resolver does not currently take into account all the packages that are installed](https://stackoverflow.com/questions/72672196/error-pips-dependency-resolver-does-not-currently-take-into-account-all-the-pa/76604141#76604141)
  ```shell
     pip install h5py
     pip install typing-extensions
     pip install wheel
  ```
- [Failed to import transformers](https://github.com/huggingface/transformers/issues/11262)
  - Try re-install
    ```shell
       conda uninstall tokenizers, transformers
       pip install transformers
    ```
