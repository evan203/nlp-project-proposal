# Code

Current status:

- can prune low ranks (w orthogonal projection) using ActSVD:
  - `cd actsvd`
  - `uv run main_low_rank_diff.py`
    - on RTX 3090, takes approx 40 mins for default model: llama-3.1-8B-instruct
  - `uv run chat.py` to test the jailbroken model
- can run difference in means:
  - `cd diff-in-means`
  - `uv run python -m pipeline.run_pipeline --model_path 'meta-llama/Llama-3.1-8B-Instruct'`
    - pipeline saves direction + completions + evaluations to `pipeline/runs/<model>/`
    - also saves a modified model with refusal direction orthogonalized out of weights
  - `uv run save_modified_model.py --model_path 'meta-llama/Llama-3.1-8B-Instruct'`
    - standalone script to apply a saved direction to the model without re-running the full pipeline
  - `cd .. && uv run chat.py --model_path diff-in-means/pipeline/runs/Llama-3.1-8B-Instruct/modified_model` to test the jailbroken model
  - if `TOGETHER_API_KEY` is set, jailbreakbench LlamaGuard2 evaluation will also run automatically
- can run cones / representational independence (cones-repind):