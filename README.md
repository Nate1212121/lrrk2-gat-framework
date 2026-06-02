# Decoding the LRRK2 G2019S Mutation: A Graph Attention Network Approach to Solving the Parkinson’s Disease GOF/LOF Paradox


**Author:** Nathan Lee


This repository contains the official codebase and pre-trained weights for the zero-shot Hybrid Graph Attention Network (GAT-MLP) designed to predict protein-protein interaction changes associated with the LRRK2 G2019S mutation.


---


## Technical Overview


By utilizing ESM-2 protein language model embeddings (3-billion parameter) and a Hybrid GAT-MLP model trained on the healthy human interactome (4,118 nodes), this framework predicts the differential link strength reorganizations (GOF/LOF directions) caused by point mutations without requiring mutant-specific training data.


---


## Getting Started


### 1. Clone the Repository
Clone the repository and navigate into the project directory:
```bash
git clone https://github.com/Nate1212121/lrrk2-gat-framework.git
cd lrrk2-gat-framework
```
### 2. Install Dependencies
Install the required packages (ensure you have PyTorch and PyTorch Geometric installed for your specific hardware):
```bash
pip3 install -r requirements.txt
```
### 3. Run Validation
To execute the end-to-end inference pipeline using our pre-computed ESM-2 embeddings and pre-trained weights (completes in < 2 minutes):
code
```bash
python3 run_full_pipeline.py
```
Note: The individual scripts in this repository are prefixed with step numbers (e.g., 01_..., 02_...) outlining the sequential research procedure from raw data ingestion to final Metascape analysis.



