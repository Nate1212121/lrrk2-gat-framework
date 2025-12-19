import os
import pandas as pd

#file paths
data_dir='./data/'
new_data_dir='../new_data/'

os.makedirs(data_dir,exist_ok=True)
os.makedirs(new_data_dir,exist_ok=True)

string_file=os.path.join(data_dir,'9606.protein.links.v12.0.txt')
score=700 #threshold for combined_score for the links (combined scores found in STRING links file)

#steps start here:
print("Step 1.1: Reading STRING links file")

string_df=pd.read_csv(string_file,sep=' ') #string dataframe from file

print(len(string_df),"links from STRING")

print("Step 1.2: Filtering links. Score threshold to be above:",score)

filtered_df=string_df[string_df['combined_score']>score] #filtered dataframe

print("Filtered links count:",len(filtered_df))

print("Step 1.3: Making edge_list.csv")

edge_output_file=os.path.join(new_data_dir,'edge_list.csv')
edge_df=filtered_df.copy()
edge_df.to_csv(edge_output_file,index=False)

print("Edge list:",edge_output_file)

print("Step 1.4: Making protein_ids.txt")

unique_proteins_output_file=os.path.join(new_data_dir,'protein_ids.txt')
unique_proteins=pd.concat([edge_df['protein1'],edge_df['protein2']]).unique()
unique_proteins_df=pd.DataFrame(unique_proteins,columns=['protein_id'])
unique_proteins_df.to_csv(unique_proteins_output_file,index=False)

print("Unique protein ids:",unique_proteins_output_file)