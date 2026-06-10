#!/bin/bash
#SBATCH -J spectroGridC
#SBATCH -p chalawan_cpu
#SBATCH -w castor4
#SBATCH -N 1
#SBATCH -n 18
#SBATCH -t 14:00:00
#SBATCH -o spectro_grid/slurm_c_%j.log
#SBATCH -e spectro_grid/slurm_c_%j.log
#
# ATLAS c-band grid.  Pipeline = run_grid.py, parameters = config_c.py.

module load hwloc/2.0.3 gnu8 openmpi3
export LD_LIBRARY_PATH=/opt/ohpc/pub/mpi/openmpi3-gnu8/4.0.2/lib:/opt/ohpc/pub/libs/hwloc/2.0.3/lib:/opt/ohpc/pub/compiler/gcc/8.3.0/lib64:$LD_LIBRARY_PATH

cd /lustre/MSSP/sittipong/buildmodule/build_ATMGRID
PY=/home/sittipong/.conda/envs/hcam-env/bin/python
CONFIG=config_c.py

# Stages A+B (models reused; SYNTHE over the c range) under MPI
/opt/ohpc/pub/mpi/openmpi3-gnu8/4.0.2/bin/mpirun -n ${SLURM_NTASKS:-20} \
    --mca pml ob1 --mca btl tcp,self \
    $PY src/run_grid.py $CONFIG

# Stage C (assemble) serially, NOT under mpirun.
$PY src/run_grid.py $CONFIG --assemble-only
