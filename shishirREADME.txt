
--> the environment to use in conda for me :: FlareGPT with python 3.10.0 ( keep this verison of python )::

Step1: Create conda environment with python 3.10.0 and activate it
conda create -n FlareGPT python=3.10.0
conda activate FlareGPT

Step2: Install requirements
pip install -r requirements.txt
pip install -r otherrequirements.txt

#for cpu : pip install llama-cpp-python==0.1.83

Step3: Run FastAPI 
uvicorn FastAPI_main_CPU:app --reload


-----------------------------------------------------------------------------------------------------------------

If The installation instructions do not work properly for Windows / CUDA systems. The following process will work:

Ensure you have the Nvidia CUDA runtime version 11.8 installed

for conda environment do this...conda install -c nvidia cuda-nvcc
else direcly download and install....
nvcc --version
Should report a CUDA version of 11.8

Create the virtual environment using conda

conda create -n FlareGPT -y
conda activate FlareGPT
conda install python=3.10 -c conda-forge -y
Verify your Python installation

python --version
Should output Python 3.10.x

Install the CUDA toolkit

conda install cudatoolkit=11.7 -c conda-forge -y
set CUDA_HOME=%CONDA_PREFIX%

Install FLAREGPT

git clone the_repo_name_from_the_github
cd FlareGPT
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117

Replace Bitsandbytes

pip uninstall bitsandbytes-windows -y
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.40.1.post1-py3-none-win_amd64.whl

Replace AutoGPTQ

pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.3.0/auto_gptq-0.3.0+cu118-cp310-cp310-win_amd64.whl


Configure FlareGPT to use GPTQ model

Open FlareChat.py and configure the model_id and model_basename

    model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
    model_basename = "gptq_model-4bit-128g.safetensors"

    there can be captial letter on model id and model basename variable used in runlocalGPT.py so check it, if so do this.

    MODEL_ID="TheBloke/Llama-2-7b-Chat-GPTQ"
    MODEL_BASENAME="model.safetensors"

    or you can use other models also like wizard vicuna and others....

Now you should be able to run EmbedVecStore.py and also run the FlareChat inference