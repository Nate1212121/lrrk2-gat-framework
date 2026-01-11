import os
import torch
import pandas as pd
from torch_geometric.data import Data

print("Step 5: Building the graph from the edge list and embeddings.")

print("\nStep 5.1: Initializing")

#checking terminal directory
print("CWD:",os.getcwd())
print("__file__:",__file__)

#file directories
new_data_dir='./new_data/'

#input files
edge_list_file=os.path.join(new_data_dir,'edge_list.csv')
normal_embeddings_file=os.path.join(new_data_dir,'normal_protein_embeddings.pt')
mutant_embedding_file=os.path.join(new_data_dir,'g2019s_protein_embedding.pt')

#output file
graph_output_file=os.path.join(new_data_dir,'ppi_pd_graph.pt')

if __name__=="__main__":
    print("\nStep 5.2: Loading data files")
    normal_embeddings=torch.load(normal_embeddings_file)

    edges_df=pd.read_csv(edge_list_file, sep=',')
    print("Edges loaded to dataframe count:",len(edges_df))
    
    print("\nStep 5.3: Filtering out invalid edges. Doing so by looking for at least one node that has an invalid embedding in the edge.")

    filtered_edges_df=edges_df[(edges_df['protein1'].isin(normal_embeddings.keys())) & (edges_df['protein2'].isin(normal_embeddings.keys()))].copy() #extra debug statement in case a protein does not have a valid embedding correlated with it

    print("Length of filtered_edges_df:",len(filtered_edges_df))

    print("\nStep 5.4: Creating Node-Index Map")

    sorted_proteins=sorted(normal_embeddings.keys()) #sorting to make sure that lrrk2 and certain other proteins are always the same accessible index. Otherwise, order could change randomly every time code is ran.
    protein_to_index={protein:i for i,protein in enumerate(sorted_proteins)}

    print("Unique node count:", len(sorted_proteins))

    print("\nStep 5.5: Building Feature Matrix X (Node Features)")

    x=torch.zeros((len(sorted_proteins),2560),dtype=torch.float) #2560 is the size of the vector for 3B model
    for protein_id,index in protein_to_index.items():
        x[index]=normal_embeddings[protein_id]

    print("\nStep 5.6: Building Edge Index")

    #changing protein names to indices
    source_nodes=filtered_edges_df['protein1'].map(protein_to_index).values
    target_nodes=filtered_edges_df['protein2'].map(protein_to_index).values

    #tensor for pytorch to have edges
    #the graph is undirected
    edge_index=torch.tensor([source_nodes,target_nodes],dtype=torch.long)
    edge_index=torch.cat([edge_index,edge_index[[1,0]]],dim=1) #making undirected by adding reverse edges
    graph_data=Data(x=x,edge_index=edge_index)

    graph_data.protein_ids=sorted_proteins
    graph_data.protein_to_index=protein_to_index

    torch.save(graph_data,graph_output_file)

    print("Final node count:", graph_data.num_nodes)
    print("Final edge count:", graph_data.num_edges)
    print("Feature matrix shape:",graph_data.x.shape)
    print("Edge index shape:",graph_data.edge_index.shape)
    print("Graph file:", graph_output_file)