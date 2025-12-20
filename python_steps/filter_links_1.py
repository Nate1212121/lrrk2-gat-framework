import os
import pandas as pd

print("Step 1.1: Initializing")

#checking directory
print("CWD:",os.getcwd())
print("__file__:",__file__)

#FILE DIRECTORIES
data_dir='./data/'
new_data_dir='./new_data/'
#directories MUST be correct for the files to be found
#(before this was ../data/ and ../new_data/ so the directories were more like /Users/nathanlee/data,
#which is not a correct directory. instead it should be /Users/nathanlee/lrrk2gcn/data/, which would be good with ./data/)

os.makedirs(data_dir,exist_ok=True)
os.makedirs(new_data_dir,exist_ok=True)
#just checking the directories are there and also add more robustness

#DATA FILES
string_link_file=os.path.join(data_dir,'9606.protein.links.v12.0.txt')
string_info_file=os.path.join(data_dir,'9606.protein.info.v12.0.txt') #translates kegg's gene names to string's homo sapien
#protein ids (9606.ENSP...) in this script as it contains both of those in its columns. the bridge for file intersection.
kegg_file=os.path.join(data_dir,'kegg_pd_protein_list.txt') #the output can only contain proteins that have at least one of
#the proteins in this file. this kegg list is made from the kegg parkinsons disease pathway proteins. this intersection-like
#filtering will be this code's second filter

score=700 #threshold for combined_score for the links (combined scores found in STRING links file)

print("\nStep 1.2: Making datatables for files")

string_link_df=pd.read_csv(string_link_file,sep=r'\s+') #string link dataframe from file with n spaces as separation
string_info_df=pd.read_csv(string_info_file,sep='\t') # STRING info file dataframe
kegg_df=pd.read_csv(kegg_file,header=None,names=['gene_name']) #kegg protein list dataframe from file

print(len(string_link_df),"interactions from STRING links")
print(len(string_info_df),"proteins from STRING info")
print(len(kegg_df),"proteins from KEGG Parkinson's Disease protein list")

print("\nStep 1.3: Converting KEGG gene names to STRING protein ids")

parkd_protein_ids_df=pd.merge(kegg_df,string_info_df,left_on='gene_name',right_on='preferred_name')
parkd_protein_ids=parkd_protein_ids_df['#string_protein_id'].unique().tolist()

print("KEGG protein list converted to STRING protein IDs. Protein list size:",len(parkd_protein_ids))

print("\nStep 1.4: Filtering links by combined_score. combined_score must be above:",score,"for the link to be kept.")

filtered_df=string_link_df[string_link_df['combined_score']>score] #filtered dataframe

print("Filtered links count after combined score filtering:",len(filtered_df))

print("\nStep 1.5: Filtering links by KEGG PD protein list")

filtered_df=filtered_df[(filtered_df['protein1'].isin(parkd_protein_ids)) | (filtered_df['protein2'].isin(parkd_protein_ids))]
print("Filtered links count after KEGG PD protein list filtering:",len(filtered_df))

print("\nStep 1.6: Making edge_list.csv")

edge_output_file=os.path.join(new_data_dir,'edge_list.csv')
edge_df=filtered_df.copy()
edge_df.to_csv(edge_output_file,index=False)

print("Edge list:",edge_output_file)

print("\nStep 1.7: Making protein_ids.txt")

unique_proteins_output_file=os.path.join(new_data_dir,'protein_ids.txt')
unique_proteins=pd.concat([edge_df['protein1'],edge_df['protein2']]).unique()
unique_proteins_df=pd.DataFrame(unique_proteins,columns=['protein_id'])
unique_proteins_df.to_csv(unique_proteins_output_file,index=False)

print("Unique protein ids:",unique_proteins_output_file)