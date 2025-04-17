# WARNING!

This is a demonstration of using the Evol Instruct pattern
The demo use case is to answer questions about business strategy.
It is not meant to be operational or investment advice.

# Evol Instruct

To replicate the example for this pattern, you will need to:

## Step 1: Download and prepare the SEC filings dataset
Run the notebook 0_download_edgar.ipynb  [might take a couple of hours]

## Step 1: Create training dataset 
Run the notebook 1_evol_instruct.ipynb  [might take 4-5 hours]

## Step 3: Sign up on HuggingFace
We will be adapter tuning the Gemma model.
You will need to accept its license on Hugging Face by clicking on the Agree and access repository button on the model page at: 
http://huggingface.co/google/gemma-3-4b-pt

## Step 4: Get a sufficiently large machine with a GPU.
We used a NVIDIA L4 with 32 GB of memory machine on Google Cloud

## Step 5: Adapter Tuning
Run the notebook 2_training.ipynb


