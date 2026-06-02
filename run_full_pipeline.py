import os
import sys
import subprocess

repo_root=os.path.dirname(os.path.abspath(__file__))
python_steps_dir=os.path.join(repo_root,'python_steps')

required_repo_files=[
    'new_data/ppi_pd_graph.pt',
    'new_data/g2019s_protein_embedding.pt',
]

steps=[
    '6_v4_train_mlp_one_layer_gat.py',
    '7_v4_predict_links.py',
]


def check_repo_files():
    missing=[]
    for rel_path in required_repo_files:
        if not os.path.isfile(os.path.join(repo_root,rel_path)):
            missing.append(rel_path)
    if missing:
        print("\nMissing files (should be in the repo):")
        for p in missing:
            print(" ",p)
        sys.exit(1)


def run_step(script_name):
    script_path=os.path.join(python_steps_dir,script_name)
    if not os.path.isfile(script_path):
        print("ERROR: script not found:",script_path)
        sys.exit(1)
    result=subprocess.run(
        [sys.executable,script_path],
        cwd=repo_root,
        check=False,
    )
    if result.returncode!=0:
        sys.exit(result.returncode)


if __name__=='__main__':
    os.chdir(repo_root)
    os.makedirs('models',exist_ok=True)
    os.makedirs('results',exist_ok=True)

    check_repo_files()

    print("Full pipeline start")

    for script_name in steps:
        run_step(script_name)

    print("\nFull pipeline complete")
    print("Outputs:")
    print("  models/gat_v4.pt")
    print("  results/v4_link_prediction_results.csv")
