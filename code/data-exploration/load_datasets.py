from huggingface_hub import login
import pandas as pd
from torch.utils.data import Dataset

hf_alpaca_path = "hf://datasets/tatsu-lab/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
hf_beaver_tails_path = "hf://datasets/PKU-Alignment/BeaverTails/round0/330k/train.jsonl.xz"

def alpaca_df(path=hf_alpaca_path):
    """Load Alpaca from HuggingFace or local path to Pandas DF"""
    assert path.endswith(".parquet"), "Alpaca path must be a .parquet file"
    return pd.read_parquet(path)

def beaver_tails_df(path=hf_beaver_tails_path):
    """Load BeaverTails from HuggingFace or local path to Pandas DF"""
    if path.endswith('.json1.xz'):
        return pd.read_json(path)
    elif path.endswith('.parquet'):
        return pd.read_parquet(path)
    raise Exception("BeaverTails path must be a .parquet file")

def load_alpaca_local(in_path=hf_alpaca_path,out_path='./data/alpaca.parquet'):
    """Download Alpaca"""
    alpaca_df = alpaca_df(in_path)
    alpaca_df.to_parquet(path=out_path)

def load_beaver_tails_local(in_path=hf_beaver_tails_path,out_path='./data/beaver_tails.parquet'):
    "Download BeaverTails"
    beaver_tails_df = beaver_tails_df(in_path)
    beaver_tails_df.to_parquet(path=out_path)

class HarmDataset(Dataset):
    def __init__(self,dataset,path=hf_alpaca_path):
        assert dataset in {'alpaca','beaver_tails'}, "dataset must be either alpaca or beaver_tails"
        self.dataset = dataset
        if dataset == 'alpaca':
            self.df = alpaca_df(path)
        if dataset == 'beaver_tails':
            self.df = beaver_tails_df(path)
            self.beaver_tails_prune_unsafe()
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.dataset == 'alpaca':
            instr = row[['instruction','input']]
            out = row['output']
        if self.dataset == 'beaver_tails':
            instr = row[['prompt','category']]
            out = row['response']
        return instr,out
    def beaver_tails_prune_unsafe(self):
        assert self.dataset == 'beaver_tails'
        self.df = self.df[not self.df['is_safe']]
        
if __name__ == "__main__":
    load_alpaca_local()
    load_beaver_tails_local()