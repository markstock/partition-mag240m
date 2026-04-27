import numpy as np
import scipy.sparse as sp
import pandas as pd
import time
import os

num_partitions = 64
out_dir = "distributed_partitions"
os.makedirs(out_dir, exist_ok=True)

print("Loading METIS partitions...")
parts = pd.read_csv('mag240m.graph.part.64', header=None, engine='c').values.squeeze()
parts = np.asarray(parts, dtype=np.int8)

# Pre-allocate a global-to-local translation array (Takes ~480MB of RAM)
# We reuse this array for every partition to save memory allocations.
global_to_local = np.empty(dataset.num_papers, dtype=np.int32)

for p in range(num_partitions):
    start = time.time()
    
    # 1. Slice the global graph for this partition
    local_nodes = np.where(parts == p)[0]
    local_adj = adj[local_nodes, :]
    
    # 2. Find all unique nodes required by this rank (Local + Halo)
    all_required_nodes = np.unique(local_adj.indices)
    
    # 3. Look up which rank owns each required node
    required_nodes_parts = parts[all_required_nodes]
    
    # 4. Create a sorting key: 
    # We force local nodes to have a key of -1 so they sort to the very front.
    # Halo nodes will keep their partition ID (0 to 63).
    sort_keys = required_nodes_parts.copy()
    sort_keys[required_nodes_parts == p] = -1
    
    # 5. Sort the required nodes based on the key
    sort_order = np.argsort(sort_keys)
    sorted_global_ids = all_required_nodes[sort_order]
    sorted_parts = sort_keys[sort_order]
    
    # 6. Build the translation map (Global ID -> Local ID)
    # Because sorted_global_ids is ordered exactly how you requested, 
    # the index in this array IS the new Local ID.
    global_to_local[sorted_global_ids] = np.arange(len(sorted_global_ids), dtype=np.int32)
    
    # 7. Translate the CSR matrix column indices to the new Local IDs
    local_adj.indices = global_to_local[local_adj.indices]
    
    # 8. CRITICAL STEP: Sort the CSR indices!
    # Because we replaced the column IDs, they are no longer strictly ascending within each row.
    # By calling sort_indices(), SciPy physically reorders the arrays in memory.
    # Because our Local IDs were assigned in rank-order, this guarantees that inside EVERY row:
    # [Local Edges] come first, then [Rank 0 Edges], [Rank 1 Edges]... etc.
    local_adj.sort_indices()
    
    # 9. Extract Metadata: How many nodes belong to each rank?
    # This tells your distributed computation exactly where the boundaries are in the arrays.
    _, counts = np.unique(sorted_parts, return_counts=True)
    part_keys = np.unique(sorted_parts)
    # Create a dictionary of {Rank: Number of Nodes}
    halo_node_counts = dict(zip(part_keys, counts))
    
    # Save the strictly ordered local adjacency matrix
    sp.save_npz(os.path.join(out_dir, f'rank_{p}_adj.npz'), local_adj)
    
    # Save the mapping so you can map Local IDs back to Global IDs if needed
    np.save(os.path.join(out_dir, f'rank_{p}_local_to_global.npy'), sorted_global_ids)
    
    # Save the halo metadata
    np.save(os.path.join(out_dir, f'rank_{p}_halo_counts.npy'), halo_node_counts)
    
    print(f"Saved Rank {p} | Local Nodes: {len(local_nodes):,} | "
          f"Total Nodes (w/ Halo): {len(sorted_global_ids):,} | "
          f"Time: {time.time()-start:.2f}s")
