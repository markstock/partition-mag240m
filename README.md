# partition-mag240m
Python scripts and workflow for partitioning the MAG240M academic publications graph

## WHAT
"MAG240M-LSC is a heterogeneous academic graph extracted from the Microsoft Academic Graph (MAG)" and you can read about it [here](https://ogb.stanford.edu/docs/lsc/mag240m/).

In some cases you may want to partition this graph for parallel processing, taking care to minimize either the number of edge cuts or the total communication between partitions. [Metis](https://ogb.stanford.edu/docs/lsc/mag240m/) is a single-threaded CPU code that is able to do this, but it needs input data in a specific format, and may take some time to run on the >1B edge MAG240m data set.

While parallel methods for graph partitioning exist (parmetis and mt-metis), they may be too difficult to build or run for a one-time task.

*This repository aims to use simple python scripts and a metis binary to partition the MAG240m data set on a login and compute node of frontier, the exascale computer at OLCF.*

## HOW
First, log on to frontier and navigate to a directory on a parallel filesystem, or otherwise ensure that you have >400 GB of disk space available. Then, on a login node (compute nodes cannot see the Internet):

    module load miniforge3
    conda create --name mag240-env python=3.13 numpy scipy pandas ogb
    source activate mag240-env
    python3 download_mag240m.py

That should complete in a half hour or so and create a directory called `dataset/mag240m_kddcup2021/`. Then get a compute node to convert this data to something metis can read:

    srun -A[projectid] -N1 -t30 -qdebug -E --pty bash
    source activate mag240-env
    python3 mag240m-to-metis.py
    exit

This should produce a 23GB file called `mag240m.graph`.

NOTE: this is as far as I have tested this workflow! If you have problems with the instructions below, please reach out and I will try to help.

Now, for the big task, actually partitioning the data set; in this case, into 64 partitions and minimizing communications volume. You need a compute node for a few hours, so you can't use the debug queue:

    module load metis
    srun -A[projectid] -N1 -t240 -pextended -n1 gpmetis -ptype=kway -objtype=vol -contig mag240m.graph 64

or just use this batch script, after changing to your project id:

    sbatch runmetis_N1n1.sh

Finally, you'll need to split, renumber (global to local IDs), and reorder (place all-local nodes earlier in the array) the metis output to allow for parallel loading and overlapping computation and communications. Note that the argument `num_partitions` must match the number given to the `gpmetis` command above.

    srun -A[projectid] -N1 -t30 -qdebug -E --pty bash
    source activate mag240-env
    python3 split-metis-partitions.py
    exit

Now you are ready to have each of the 64 tasks in your primary compute job load only their partition of the data. That code might look something like this:

```
# Code running on Rank 'rank_id'
import numpy as np
import scipy.sparse as sp

rank_id = 4 # Example

# Load the local graph structure
local_adj = sp.load_npz(f"distributed_partitions/rank_{rank_id}_adj.npz")

# Load the global node IDs that this rank owns
global_node_ids = np.load(f"distributed_partitions/rank_{rank_id}_nodes.npy")

print(f"Rank {rank_id} loaded and ready. Using {local_adj.data.nbytes / 1e6:.2f} MB of edge data.")
```


