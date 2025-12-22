import os
import pandas as pd
import json

print("Step 2: Getting sequences for proteins in protein_ids list (nodes list) so they can be later enriched with help of ESM-2 language model.")

print("\nStep 2.1: Initializing")

#checking terminal directory
print("CWD:",os.getcwd())
print("__file__:",__file__)

#FILE DIRECTORIES
data_dir='./data/'
new_data_dir='./new_data/'

nodes_file=os.path.join(new_data_dir,'protein_ids.txt')
string_sequence_file=os.path.join(data_dir,'9606.protein.sequences.v12.0.fa')

print("\nStep 2.2: Making dataframe and list for nodes from protein_ids.txt")

node_df=pd.read_csv(nodes_file,header=None,names=['protein_id'])
node_list=node_df['protein_id'].values

print("Node/protein id count:",len(node_df)-1) #excluding header row

print("\nStep 2.3: Making map with key being STRING protein id and value being STRING protein sequence")

string_sequence_map={}
current_id=None
current_sequence=[]

with open(string_sequence_file,'r') as f:
    for line in f:
        if line.startswith('>'):
            if current_id and current_id in node_list:
                string_sequence_map[current_id]=''.join(current_sequence)
            header_id=line.strip()[1:]
            current_id=header_id
        current_sequence=[]
    else:
        current_sequence.append(line.strip())

if current_id and current_id in node_list:
    string_sequence_map[current_id]=''.join(current_sequence) # updating map for last protein id

print("\n Sequences mapped count:",len(string_sequence_map))

print("\nStep 2.4: Making output file with sequence map")

sequences_output_file=os.path.join(new_data_dir,'protein_sequences.json')
with open(sequences_output_file,'w') as f:
    json.dump(string_sequence_map,f)

print("Protein sequences saved to:",sequences_output_file)