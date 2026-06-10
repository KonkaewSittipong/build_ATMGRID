#!/bin/bash
#SBATCH -J spectroAtlas
#SBATCH -p chalawan_gpu
#SBATCH -w pollux2
#SBATCH -N 1
#SBATCH -n 18
#SBATCH -t 14:00:00
#SBATCH -o spectro_grid/slurm_atlas_%j.log
#SBATCH -e spectro_grid/slurm_atlas_%j.log
#
# ATLAS c-band grid.  Pipeline = run_grid.py, parameters = config_atlas.py.

module load hwloc/2.0.3 gnu8 openmpi3
export LD_LIBRARY_PATH=/opt/ohpc/pub/mpi/openmpi3-gnu8/4.0.2/lib:/opt/ohpc/pub/libs/hwloc/2.0.3/lib:/opt/ohpc/pub/compiler/gcc/8.3.0/lib64:$LD_LIBRARY_PATH

cd /lustre/MSSP/sittipong/buildmodule/build_ATMGRID
PY=/home/sittipong/.conda/envs/hcam-env/bin/python
CONFIG=config_atlas.py

# Stages A+B (models reused; SYNTHE over the c range) under MPI
/opt/ohpc/pub/mpi/openmpi3-gnu8/4.0.2/bin/mpirun -n ${SLURM_NTASKS:-20} \
    --mca pml ob1 --mca btl tcp,self \
    $PY src/run_grid.py $CONFIG

# Stage C (assemble) serially, NOT under mpirun.
$PY src/run_grid.py $CONFIG --assemble-only
