# The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence

This repository contains the code for the experiments presented in our paper: [The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence](https://arxiv.org/abs/2502.17420).

## Overview

The code is primarily designed for VSCode interactive mode (similar to Jupyter notebooks) but also mostly supports command-line execution. We use [Weights & Biases](https://wandb.ai/) for experiment logging and result storage.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/wollschlager/topdown-interpretability.git
    cd topdown-interpretability
    ```

2.  **Set up the environment:**

    ```bash
    # Create the conda environment
    conda create -n rdo python=3.10
    conda activate rdo
    pip install -r requirements.txt
    ```

## Configuration (.env)

This project uses a `.env` file in the root directory to manage configuration settings. You should take a look at this file and change the parameters to suit your needs. For reference, you can have a look at the .env_example file

## Usage

### Computing DIM Directions

   *   Navigate to the `refusal_direction` subdirectory.
   *   Run the pipeline script:
       ```bash
       cd refusal_direction
       python -m pipeline.run_pipeline --model_path google/gemma-2-2b-it
       cd ..
       ```
     *(Note: This part has minimal differences from [https://github.com/andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction))*

**2. Compute the Refusal Direction Optimization (RDO) directions, cones, or rep. ind. directions:**

   *   Train the RDO direction:

       ```bash
       python rdo.py --train_direction --model google/gemma-2-2b-it
       ```
   *   Train a direction orthogonal to the main RDO direction:

       ```bash
       python rdo.py --train_orthogonal_direction --model google/gemma-2-2b-it
       ```
   *   Train a direction independent from the DIM direction:

       ```bash
       python rdo.py --train_independent_direction --model google/gemma-2-2b-it
       ```
   *   Train a refusal cone (example with dimension 2):

       ```bash
       # Adjust min/max_cone_dim, n_sample as needed
       python rdo.py --train_cone --min_cone_dim 2 --max_cone_dim 2 --n_sample 16 --model google/gemma-2-2b-it
       ```

### Evaluating Directions

Use the `run_rdo_pipeline.py` script to evaluate the computed RDO, Cone, or other directions stored in a Weights & Biases run.

*   **Evaluate a specific W&B run (e.g., RDO direction):**

    ```bash
    # Replace 'fast-shadow-11' with your W&B run display name
    python -m pipeline.run_rdo_pipeline --wandb_run fast-shadow-11
    ```
*   **Evaluate the basis vectors of a refusal cone (e.g., dimension 2):**

    ```bash
    # Replace 'dim_2' with your W&B run display name corresponding to the cone
    python -m pipeline.run_rdo_pipeline --wandb_run dim_2
    ```
*   **After evaluating the basis vectors, you can evaluate the performance of Monte-Carlo samples in the cone:**

    ```bash
    # Replace 'dim_2' with your W&B run display name corresponding to the cone
    python -m pipeline.run_rdo_samples --wandb_run dim_2 --sample_start_idx 0 --sample_end_idx 8 # evaluates sample 0 to 7, by default 512 samples are saved during basis vector evaluation
    ```

## Analysis

The `cosinesims_analysis.py` script can be used to analyze the cosine similarities between the computed DIM direction and rep. ind. directions.

### Comparing Jailbroken Models

This repository now includes tools for computing **Representational Independence (RepInd) Statistics** between jailbroken models created by different methodologies:

**Supported jailbreak methods:**
- **ActSVD**: Activation subspace modification via low-rank decomposition (`actsvd/`)
- **Diff-in-Means**: Orthogonalization against refusal directions (`diff-in-means/`)

#### 1. Extract DIM Directions from Jailbroken Models

First, extract DIM directions from your saved jailbroken models:

```bash
python extract_dim_directions.py \
    --model_path /path/to/actsvd_jailbroken_model \
    --harmful_prompts data/saladbench_splits/harmful_train.json \
    --harmless_prompts data/saladbench_splits/harmless_train.json \
    --n_samples 500 \
    --output_dir results/directions
```

Repeat for the diff-in-means jailbroken model.

#### 2. Compute RepInd Statistics Between Models

Compare two jailbroken models to analyze their representational independence:

```bash
python jailbreak_comparison.py \
    --model_a_path /path/to/actsvd_jailbroken_model \
    --model_b_path /path/to/diff_in_means_jailbroken_model \
    --data_dir data/saladbench_splits \
    --n_samples 500 \
    --output_dir results/jailbreak_comparison
```

**Output includes:**
- `repind_results.json`: Complete statistics including:
  - `repind_mse`: RepInd MSE (lower = more independent subspaces)
  - `repind_mae`: RepInd MAE
  - `direction_similarity`: Cosine similarity between DIM directions
  - `layer_diffs`: Per-layer contribution to RepInd
- `repind_visualization.png`: Visual summary of the analysis

#### Understanding RepInd Statistics

The RepInd statistic (based on the Cones paper) measures how independent two jailbreak subspaces are by computing:

```
RepInd = MSE[(cos_sim_A(A) - cos_sim_A(B)) + (cos_sim_B(B) - cos_sim_B(A))]
```

Where:
- `cos_sim_A(X)` = cosine similarity of direction from model A with activations from model X
- `cos_sim_B(X)` = cosine similarity of direction from model B with activations from model X

**Interpretation:**
- **Low RepInd (~0.01-0.05)**: Subspaces are highly independent; different mechanisms for jailbreaking
- **Medium RepInd (~0.05-0.15)**: Partial overlap in jailbreaking mechanisms
- **High RepInd (~0.15+)**: Subspaces overlap significantly; similar jailbreaking mechanisms

#### API Usage

You can also use the functions programmatically:

```python
from transformer_utils import (
    load_model_and_tokenizer,
    get_residual_stream_activations,
    compute_mean_diff_direction,
)
from jailbreak_comparison import (
    compute_repind_statistic,
    compute_direction_similarity,
    compute_layer_wise_cosine_similarities,
)

# Load models and extract activations
model_a, tokenizer_a = load_model_and_tokenizer(model_a_path)
model_b, tokenizer_b = load_model_and_tokenizer(model_b_path)

activations_a = get_residual_stream_activations(model_a, tokenizer_a, prompts)
activations_b = get_residual_stream_activations(model_b, tokenizer_b, prompts)

# Compute directions
direction_a = compute_mean_diff_direction(activations_a, activations_a, layer=best_layer)
direction_b = compute_mean_diff_direction(activations_b, activations_b, layer=best_layer)

# Compute statistics
repind_stats = compute_repind_statistic(activations_a, activations_b, direction_a, direction_b)
dir_sim = compute_direction_similarity(direction_a, direction_b)
```
