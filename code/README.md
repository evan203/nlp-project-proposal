# Code

Current status:

- can prune low ranks (w orthogonal projection) using ActSVD:
  - `cd actsvd`
  - `uv run main_low_rank_diff.py`
    - on RTX 3090, takes approx 40 mins for default model: llama-3.1-8B-instruct
  - `uv run chat.py` to test the jailbroken model
- can run difference in means:
  - `cd diff-in-means`
  - `uv run python -m pipeline.run_pipeline --model_path 'meta-llama/Llama-3.1-8B-Instruct`
  - `TOGETHER_API_KEY` environment variable must be set to run jailbreakbench evaluation
