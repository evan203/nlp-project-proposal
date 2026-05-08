import matplotlib.pyplot as plt
import seaborn as sns
import torch

# gemma
# df = [0.0000, 0.0169, 0.2496, 0.1817, 0.3434, 0.3361, 0.4360, 0.6132, 0.6126, 0.6201, 0.7386, 0.7186, 0.6955, 0.6728, 0.6680, 0.6482, 0.6515, 0.6340]
# llama
df = [
    0.0000,
    0.4551,
    0.2564,
    0.1958,
    0.3240,
    0.3545,
    0.3442,
    0.3892,
    0.4471,
    0.5384,
    0.5770,
    0.7126,
    0.7303,
    0.7196,
    0.6878,
    0.6792,
    0.6852,
    0.6852,
    0.6765,
    0.6797,
    0.6721,
    0.6554,
    0.6326,
    0.6117,
    0.5967,
    0.5918,
    0.5803,
    0.5694,
    0.5545,
    0.5405,
    0.5140,
    0.5240,
]

g = sns.barplot(x=[f"Layer {i}" for i in range(len(df))], y=df)
g.set_title("Llama DIM similarities for reference dataset vs twinprompt")
g.set_ylabel("Cosine Similarity")
plt.xticks(rotation=30)
plt.show()
