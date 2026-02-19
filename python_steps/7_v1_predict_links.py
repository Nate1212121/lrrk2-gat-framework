import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from scipy import stats
from scipy.stats import wilcoxon

print("Step 7: Running GCNs and finding differences in normal vs mutant LRRK2")

print("\nStep 7.1: Initializing")

#checking terminal directory
print("CWD:",os.getcwd())
print("__file__:",__file__)

#directories
new_data_dir='./new_data/'
models_dir='./models/'
results_dir='./results/'

#files
graph_file=os.path.join(new_data_dir, 'ppi_pd_graph.pt')
gcn_v1_file=os.path.join(models_dir, 'gcn_v1.pt')
gcn_v2_file=os.path.join(models_dir, 'gcn_v2.pt')
gcn_v3_file=os.path.join(models_dir, 'gcn_v3.pt')
mutant_embedding_file=os.path.join(new_data_dir, 'g2019s_protein_embedding.pt')
results_output_file=os.path.join(results_dir, 'v1_link_prediction_results.csv')

#out_channels=32
out_channels=64 
#out_channels=128
#out_channels=512

class link_predictor(torch.nn.Module):
    #message passer function (structure of the layers)
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.conv1=GCNConv(in_channels, out_channels)

    #encoder (performing the message passing)
    def encode(self,x,edge_index):
        x=self.conv1(x,edge_index).relu() #passing through layer and adding non-linearity with relu
        return x

    #decoder
    def decode(self, z,edge_label_index): #dot product (often used for ppi link predictions)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=1) 


if __name__=="__main__":
    if torch.backends.mps.is_available():
        device=torch.device('mps')
    else:
        device=torch.device('cpu')
    
    graph_data=torch.load(graph_file,weights_only=False).to(device)

    mutant_embedding=torch.load(mutant_embedding_file,map_location=device)

    model=link_predictor(
        in_channels=graph_data.num_features,
        out_channels=out_channels
    ).to(device)

    model.load_state_dict(torch.load(gcn_v1_file,map_location=device))
    model.eval()

    lrrk2_idx=graph_data.protein_to_index['9606.ENSP00000298910']

    source_row=torch.full((graph_data.num_nodes,),fill_value=lrrk2_idx,dtype=torch.long)
    target_row=torch.arange(graph_data.num_nodes,dtype=torch.long)

    prediction_edges=torch.stack([source_row,target_row],dim=0)
    print("\nStep 7.2: Calculating scores for normal LRRK2")
     
    encoded_normal=model.encode(graph_data.x,graph_data.edge_index)
    decoded_normal=model.decode(encoded_normal,prediction_edges)

    print("\nStep 7.3: Cloning graph for G2019S LRRK2 and calculating scores")
    #MUTATED GRAPH MAKING
    mutant_graph_data=graph_data.clone()
    mutant_graph_data.x[lrrk2_idx]=mutant_embedding #replacing normal lrrk2 with g2019s lrrk2 embedding

    encoded_mutant=model.encode(mutant_graph_data.x,mutant_graph_data.edge_index)
    decoded_mutant=model.decode(encoded_mutant,prediction_edges)

    score=(decoded_mutant-decoded_normal)
    print("\nStep 7.4: Saving results")
    id_dict={}
    for name, i in graph_data.protein_to_index.items():
        id_dict[i]=name

    protein_names=[]
    for i in range(graph_data.num_nodes):
        protein_names.append(id_dict[i])
    
    results_df=pd.DataFrame()
    results_df['protein_id']=protein_names
    results_df['normal_score']=decoded_normal.detach().cpu().numpy()#to disconnect it from the neural network graph weights we have detach
    results_df['mutant_score']=decoded_mutant.detach().cpu().numpy()#to disconnect it from the neural network graph weights we have detach
    results_df['final_score']=score.detach().cpu().numpy()#to disconnect it from the neural network graph weights we have detach

    results_df=results_df.sort_values(by='final_score',ascending=False)
    results_df.to_csv(results_output_file,index=False)

    gof_scores=results_df[results_df['final_score']>0]['final_score']
    lof_scores=results_df[results_df['final_score']<0]['final_score']

    statistic, p_value=wilcoxon(results_df['final_score'])
    print(f"\nWilcoxon signed-rank test on final scores: statistic={statistic}, p-value={p_value}")