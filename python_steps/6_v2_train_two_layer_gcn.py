import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
import numpy as np
from sklearn.metrics import roc_auc_score

print("Step 6: Training 2-layer GCN")

print("\nStep 6.1: Initializing")

#checking terminal directory
print("CWD:",os.getcwd())
print("__file__:",__file__)

new_data_dir='./new_data/'
models_dir='./models/'

graph_file=os.path.join(new_data_dir, 'ppi_pd_graph.pt')
model_output_file=os.path.join(models_dir, 'gcn_v2.pt')

#variable variables
#epochs=50 
epochs=100
#epochs=500 

learning_rate=0.001
# learning_rate=0.005
# learning_rate=0.01
#learning_rate=0.1

# hidden_channels_one=128
hidden_channels_one=256
# hidden_channels_one=512

#out_channels=32
out_channels=64
#out_channels=128
#out_channels=512


print("\nEpochs:", epochs)
print("Learning rate:",learning_rate)
print("Out channel count:",out_channels)

class link_predictor(torch.nn.Module):
    #message passer function (structure of the layers)
    def __init__(self,in_channels, hidden_channels_one,out_channels):
        super().__init__()
        self.conv1=GCNConv(in_channels, hidden_channels_one)
        self.conv2=GCNConv(hidden_channels_one,out_channels)#second layer

    #encoder (performing the message passing)
    def encode(self,x,edge_index):
        x=self.conv1(x,edge_index).relu() #passing through layer and adding non-linearity with relu
        x=self.conv2(x,edge_index).relu()
        return x

    #decoder
    def decode(self, z,edge_label_index): #dot product for ppi link predictions
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=1) 

def train(model,data,optimizer,criterion):
    model.train()
    optimizer.zero_grad()
    z=model.encode(data.x,data.edge_index)
    
    edge_label_index=torch.cat([data.pos_edge_label_index, data.neg_edge_label_index],dim=1)
    edge_label=torch.cat([torch.ones(data.pos_edge_label_index.size(1)),torch.zeros(data.neg_edge_label_index.size(1))],dim=0)
    
    out=model.decode(z,edge_label_index).view(-1)
    loss=criterion(out, edge_label.to(out.device))
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test(model, data):
    model.eval()
    z=model.encode(data.x,data.edge_index)

    edge_label_index=torch.cat([data.pos_edge_label_index, data.neg_edge_label_index], dim=1)
    edge_label=torch.cat([torch.ones(data.pos_edge_label_index.size(1)), torch.zeros(data.neg_edge_label_index.size(1))], dim=0)

    out=model.decode(z, edge_label_index).view(-1)
    return roc_auc_score(edge_label.cpu().numpy(), out.cpu().numpy())

if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device=torch.device('mps')
    else:
        device=torch.device('cpu')
    #mps or cpu
    #mps for mac

    print("\nStep 6.2: Loading graph.")
    graph_data=torch.load(graph_file, weights_only=False)
    print("Graph data node count:", graph_data.num_nodes)

    print("\nStep 6.3: Splitting data+Initializing model.")
    transform=RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        split_labels=True,
    )
    train_data,val_data,test_data = transform(graph_data)

    model=link_predictor(
        in_channels=graph_data.num_features,
        hidden_channels_one=hidden_channels_one,
        out_channels=out_channels
    ).to(device)

    optimizer=torch.optim.Adam(params=model.parameters(),lr=learning_rate)
    criterion=torch.nn.BCEWithLogitsLoss()

    print("\nStep 6.4: Training and testing model (going through epochs)")
    highest_auc=0
    for i in range(1,epochs+1):
        loss=train(model, train_data.to(device),optimizer,criterion)
        auc=test(model,val_data.to(device))
        
        if(auc>highest_auc):
            highest_auc=auc
            torch.save(model.state_dict(), model_output_file)
            
        if(i%10==0):
            print("Epoch:",i)
            print("Loss:",loss.item())
            print("AUC:",auc,"\n")
            
    print("Best training AUC (for validation epochs):",highest_auc)
    
    print("\nStep 6.5: Loading results and saving model.")

    model.load_state_dict(torch.load(model_output_file, weights_only=True))
    final_auc=test(model, test_data.to(device))
    
    print("\nAUC for final test dataset:",final_auc)
    print("\nModel file saved in:", model_output_file)