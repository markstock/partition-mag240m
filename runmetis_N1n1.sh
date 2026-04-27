#!/bin/bash
#SBATCH -A ven114
#SBATCH -J metis-mag240m
#SBATCH -o runmetis_%j.out
#SBATCH -t 240
#SBATCH -N 1
#SBATCH -p extended

unset SLURM_EXPORT_ENV

# Load modules
module reset
module load metis/5.1.0
module list

export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

# split into this many groups
NUM_PARTS=(8 64)

echo "======================================================================"
echo "METIS graph partitioning for optimized communication"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================================"

# Test multiple cross-cluster edge densities
for NUM_PARTITIONS in "${NUM_PARTS[@]}"; do

    echo "======================================================================"
    echo "Test: Split into $NUM_PARTITIONS partitions"
    echo "======================================================================"

    DATETIMESTAMP=$(date +'%Y-%m-%d %H:%M:%S.%6N')
    echo "Rank 0 time before calling srun: $DATETIMESTAMP"

    # Run test and capture output, with error handling to continue on failure
    srun -n1 -c7 \
        gpmetis -ptype=kway -objtype=vol -contig mag240m.graph $NUM_PARTITIONS

    DATETIMESTAMP=$(date +'%Y-%m-%d %H:%M:%S.%6N')
    echo "Rank 0 time after calling srun: $DATETIMESTAMP"
    echo ""

done

echo ""
echo "======================================================================"
echo "All complete!"
echo "======================================================================"
