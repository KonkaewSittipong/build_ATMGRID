#!/bin/bash
#SBATCH -J spectroRetry
#SBATCH -p chalawan_gpu
#SBATCH -w pollux3
#SBATCH -N 1
#SBATCH -n 14
#SBATCH -t 2:00:00
#SBATCH -o spectro_grid/slurm_retry_%j.log
#SBATCH -e spectro_grid/slurm_retry_%j.log
#
# Retry the 13 ATLAS12 models that did not converge in the main grid
# (6500-8250 K, A/F-type), with stronger settings: 60 iterations + a 2-step
# ladder from the starting model. Then SYNTHE + re-assemble the o grid.
# Resume-safe: the 274 already-built models/spectra are skipped.

module load hwloc/2.0.3 gnu8 openmpi3

PYTHON=/home/sittipong/.conda/envs/hcam-env/bin/python
MPIRUN=/opt/ohpc/pub/mpi/openmpi3-gnu8/4.0.2/bin/mpirun
export LD_LIBRARY_PATH=/opt/ohpc/pub/mpi/openmpi3-gnu8/4.0.2/lib:/opt/ohpc/pub/libs/hwloc/2.0.3/lib:/opt/ohpc/pub/compiler/gcc/8.3.0/lib64:$LD_LIBRARY_PATH

BASE=/lustre/MSSP/sittipong/buildmodule/build_ATMGRID
ATLAS=/lustre/MSSP/sittipong/packages/ATLAS
MODELS=$BASE/spectro_grid/models
H5DIR=$BASE/spectro_grid/h5
OUTGRID=$BASE/spectro_grid/ATLAS9.AB.Atlas.o.h5
cd "$BASE"

N=${SLURM_NTASKS:-14}
MPIFLAGS="--mca pml ob1 --mca btl tcp,self"

# Unique Teff/logg covering the 13 failed nodes (product = 56 combos; the 43
# already-converged ones are skipped, only the 13 missing are rebuilt).
TEFF="6500 6750 7000 7250 7500 7750 8000 8250"
LOGG="2.5 3.0 3.5 4.0 4.5 5.0 5.5"
FEH="0.0"
WSTART=5400
WEND=8400
RESOLU=100000

echo "=================================================================="
echo " RETRY STAGE A — 13 failed ATLAS12 models (iter 60, steps 2)"
echo "=================================================================="
$MPIRUN -n $N $MPIFLAGS $PYTHON $ATLAS/run_atlas12.py \
    --teff $TEFF --logg $LOGG --feh $FEH \
    --vturb 2.0 --iter 60 --steps 2 --clean --outdir "$MODELS"

echo "=================================================================="
echo " RETRY STAGE B — SYNTHE surface intensity for newly-built models"
echo "=================================================================="
$MPIRUN -n $N $MPIFLAGS $PYTHON $BASE/spectro_grid_synthe_mpi.py \
    --grid-dir "$MODELS" --specdir "$H5DIR" \
    --wstart $WSTART --wend $WEND --resolu $RESOLU

echo "=================================================================="
echo " RETRY STAGE C — re-assemble ATLAS o photometry grid"
echo "=================================================================="
$PYTHON $BASE/assemble_grid.py \
    --specdir "$H5DIR" \
    --filter "$BASE/filters/Misc_Atlas.o.txt" \
    --filter-name o --filter-desc "ATLAS o" \
    --ext 0.846 --out "$OUTGRID"

echo "DONE -> $OUTGRID"
