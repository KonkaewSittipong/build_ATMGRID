#!/bin/bash
#SBATCH -J atlasMBtest
#SBATCH -p chalawan_gpu
#SBATCH -w pollux2
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -t 01:00:00
#SBATCH -o spectro_grid_test/slurm_mbtest_%j.log
#SBATCH -e spectro_grid_test/slurm_mbtest_%j.log
#
# Small multi-band (c+o) validation. Pipeline=run_grid.py, params=config_atlas_test.py.

module load hwloc/2.0.3 gnu8 openmpi3
export LD_LIBRARY_PATH=/opt/ohpc/pub/mpi/openmpi3-gnu8/4.0.2/lib:/opt/ohpc/pub/libs/hwloc/2.0.3/lib:/opt/ohpc/pub/compiler/gcc/8.3.0/lib64:$LD_LIBRARY_PATH

cd /lustre/MSSP/sittipong/buildmodule/build_ATMGRID
mkdir -p spectro_grid_test
PY=/home/sittipong/.conda/envs/hcam-env/bin/python
CONFIG=config_atlas_test.py

# Stages A+B under MPI
/opt/ohpc/pub/mpi/openmpi3-gnu8/4.0.2/bin/mpirun -n ${SLURM_NTASKS:-5} \
    --mca pml ob1 --mca btl tcp,self \
    $PY src/run_grid.py $CONFIG

# Stage C (assemble) serially
$PY src/run_grid.py $CONFIG --assemble-only
