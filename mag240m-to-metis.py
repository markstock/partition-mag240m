import os
import torch

# ==========================================
# FIX FOR PYTORCH 2.6+ AND OGB INCOMPATIBILITY
# Force torch.load to use the old behavior 
# so OGB can unpickle its NumPy dictionaries.
# ==========================================
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# ==========================================

import numpy as np
import scipy.sparse as sp
from ogb.lsc import MAG240MDataset
import time

print("Loading dataset...")
dataset = MAG240MDataset(root='dataset/')
edge_index = dataset.edge_index('paper', 'cites', 'paper')
num_papers = dataset.num_papers

print("Building sparse matrix...")
# Use int8 for the data array to save a bit of memory overhead
data = np.ones(edge_index.shape[1], dtype=np.int8)
adj = sp.coo_matrix((data, (edge_index[0], edge_index[1])), 
                    shape=(num_papers, num_papers))

print("Symmetrizing graph (METIS requires undirected)...")
adj = adj + adj.T 
adj.setdiag(0)         # Remove self-loops
adj.eliminate_zeros()  # Clean up the matrix
adj = adj.tocsr()      # Convert to CSR for O(1) row lookups

# The number of undirected edges is half the non-zeros
num_undirected_edges = adj.nnz // 2

print(f"Graph ready. Nodes: {num_papers}, Edges: {num_undirected_edges}")
print("Writing to METIS format (This will take ~10 minutes)...")

start_time = time.time()

# Extract the raw NumPy arrays from the CSR matrix for faster iteration
indptr = adj.indptr
indices = adj.indices

# Open with an 8MB buffer for high-speed disk write
with open("mag240m.graph", "w", buffering=8388608) as f:
    f.write(f"{num_papers} {num_undirected_edges}\n")
    
    for i in range(num_papers):
        # METIS is 1-indexed, so we add 1
        neigh = indices[indptr[i] : indptr[i+1]] + 1
        
        # If a node is isolated, this naturally writes an empty line, 
        # which is the correct METIS specification.
        f.write(" ".join(map(str, neigh)) + "\n")

print(f"Done! File written in {(time.time() - start_time) / 60:.2f} minutes.")
