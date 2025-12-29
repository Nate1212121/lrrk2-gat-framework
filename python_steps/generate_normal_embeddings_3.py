import os
import time
import requests
import pandas as pd
import torch
from transformers import EsmTokenizer, EsmModel
from tqdm import tqdm #Using tqdm for progress bars for longer tasks
import numpy as np
import json

print("Step 3: Generating embeddings for normal proteins using ESM-2 language model (WITH SLIDING WINDOW)(not G2019S LRRK2 mutated protein)")

print("\nStep 3.1: Initializing")

#checking terminal directory
print("CWD:",os.getcwd())
print("__file__:",__file__)

#FILE DIRECTORIES
new_data_dir='./new_data/'

#INPUT FILES
nodes_file=os.path.join(new_data_dir,'protein_ids.txt')
sequences_file=os.path.join(new_data_dir,'protein_sequences.json')

esm2_model='facebook/esm2_t33_650M_UR50D' #ESM2 model name

print("\nStep 3.2: Loading protein sequences")

with open(sequences_file,'r' ) as f:
    sequence_map=json.load(f)

print("Loaded protein sequences count:",len(sequence_map))

#Defining function to generate embeddings for better readability and debugging purposes
def generate_embeddings(sequences,model_name):
    if torch.backends.mps.is_available():
        device=torch.device('mps')
    else:
        device=torch.device('cpu')
    #mps is for mac; faster than cpu if available
    
    tokenizer=EsmTokenizer.from_pretrained(model_name)
    model=EsmModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings_map={}
    for protein_id,sequence in tqdm(sequences.items(),desc="Generating embeddings"):
        index=0
        temp_embedding=[]
        while index<len(sequence):
            end=index+1022
            if end>len(sequence):
                end=len(sequence)
            window=sequence[index:end]
            if len(window)>1022:
                window=window[:1022]  # ESM2 token limit is 1024, but two are special cases (beginning and end)
            inputs=tokenizer(window,return_tensors='pt',add_special_tokens=True).to(device)
            with torch.no_grad():
                outputs=model(**inputs)
            last_hidden_state=outputs.last_hidden_state
            embedding=last_hidden_state[0,1:-1].mean(dim=0)
            temp_embedding.append(embedding.cpu())
            index+=511  # sliding window with overlap of 511 amino acids
        embeddings_map[protein_id]=torch.stack(temp_embedding).mean(dim=0)
    return embeddings_map

print("\nStep 3.3: Generating and saving normal protein embeddings to file")

print("Using ESM-2 model:",esm2_model)

if __name__=='__main__':
    output_embeddings_file=os.path.join(new_data_dir,'normal_protein_embeddings.pt') #output file format for protein embeddings is .pt when using pytorch
    normal_embeddings=generate_embeddings(sequence_map,esm2_model)
    torch.save(normal_embeddings,output_embeddings_file)

    print("Normal protein embeddings saved to:",output_embeddings_file)


#finished running, run time took less than 2 mins