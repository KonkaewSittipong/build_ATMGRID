#!/bin/bash
#SBATCH -J spectroTest
#SBATCH -p chalawan_gpu
#SBATCH -N 1
#SBATCH -n 14
#SBATCH -t 2:00:00
#SBATCH -o test_grid/slurm_%j.log
#SBATCH -e test_grid/slurm_%j.log
#
# Small test grid.  Pipeline = run_grid.py, parameters = config_test.py.

module load hwloc/2.0.3 gnu8 openmpi3
export LD_LIBRARY_PATH=/opt/ohpc/pub/mpi/openmpi3-gnu8/4.0.2/lib:/opt/ohpc/pub/libs/hwloc/2.0.3/lib:/opt/ohpc/pub/compiler/gcc/8.3.0/lib64:$LD_LIBRARY_PATH

cd /lustre/MSSP/sittipong/buildmodule/build_ATMGRID
PY=/home/sittipong/.conda/envs/hcam-env/bin/python
CONFIG=config_test.py

/opt/ohpc/pub/mpi/openmpi3-gnu8/4.0.2/bin/mpirun -n ${SLURM_NTASKS:-14} \
    --mca pml ob1 --mca btl tcp,self \
    $PY src/run_grid.py $CONFIG

$PY src/run_grid.py $CONFIG --assemble-only
