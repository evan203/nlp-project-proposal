from huggingface_hub import login
from os import getcwd, mkdir
from os.path import exists, join
import pandas as pd

def load_datasets(path = join(getcwd(),'data'), exclude_datasets = []):
    """Load training datasets - AdvBench, HelpSteer, TruthfulQA, Alpaca.
    
    Optional args:
        path (path-like object): Path to download dataset CSV files. Defaults to ./data/
        exclude_datasets (container of str): dataset names to exclude (adv_bench, help_steer, truthful_qa, alpaca)
    """
    login()

    if not exists(path):
        mkdir(path)


    if 'adv_bench' not in exclude_datasets:
        adv_bench = pd.read_parquet("hf://datasets/walledai/AdvBench/data/train-00000-of-00001.parquet")
        adv_bench.to_csv(join(path,'adv_bench.csv'))
    if 'help_steer' not in exclude_datasets:
        help_steer = pd.read_json("hf://datasets/nvidia/HelpSteer/train.jsonl.gz", lines=True)
        help_steer.to_csv(join(path,'help_steer.csv'))
    if 'truthful_qa' not in exclude_datasets:
        truthful_qa = pd.read_csv("hf://datasets/domenicrosati/TruthfulQA/train.csv")
        truthful_qa.to_csv(join(path,'truthful_qa.csv'))
    if 'alpaca' not in exclude_datasets:
        alpaca = pd.read_parquet("hf://datasets/tatsu-lab/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
        alpaca.to_csv(join(path,'alpaca.csv'))


if __name__ == "__main__":
    load_datasets()

