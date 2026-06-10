#!/bin/bash
#SBATCH -J spectroGrid
#SBATCH -p chalawan_gpu
#SBATCH -w pollux3
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -t 4:00:00
#SBATCH -o spectro_grid/slurm_%j.log
#SBATCH -e spectro_grid/slurm_%j.log
#
# Full 41 x 7 ATLAS9 specific-intensity grid -> ATLAS o photometry grid.
#   Stage A : ATLAS12 models        (mpirun run_atlas12.py, atomic-only)
#   Stage B : SYNTHE surface I(mu)   (mpirun spectro_grid_synthe_mpi.py, 5400-8400 A)
# NOTE: 5400-8400 A covers the o band (5550-8270) and runs ~14 s/node. Do NOT
#       extend blue to ~4000 A (needed for c band): the 4000-5400 region is so
#       line-dense that synthe.exe takes ~6 h/node. The c grid needs a separate
#       strategy (lower R and/or trimmed line list).
#   Stage C : band-integrate o       (serial assemble_grid.py, ext=0.846)
# Resume-safe: completed models / node spectra are skipped on re-submit.

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
mkdir -p "$MODELS" "$H5DIR"

N=${SLURM_NTASKS:-28}
MPIFLAGS="--mca pml ob1 --mca btl tcp,self"

# ── grid axes ───────────────────────────────────────────────────────────────
TEFF="3500 3750 4000 4250 4500 4750 5000 5250 5500 5750 6000 6250 6500 6750 7000 7250 7500 7750 8000 8250 8500 8750 9000 9250 9500 9750 10000 10250 10500 10750 11000 11250 11500 11750 12000 12250 12500 12750 13000 14000 15000"
LOGG="2.5 3.0 3.5 4.0 4.5 5.0 5.5"
FEH="0.0"
WSTART=5400
WEND=8400
RESOLU=100000

echo "=================================================================="
echo " STAGE A — ATLAS12 models (41 x 7 = 287)"
echo "=================================================================="
$MPIRUN -n $N $MPIFLAGS $PYTHON $ATLAS/run_atlas12.py \
    --teff $TEFF --logg $LOGG --feh $FEH \
    --vturb 2.0 --iter 45 --clean --outdir "$MODELS"

echo "=================================================================="
echo " STAGE B — SYNTHE surface intensity I(mu),  $WSTART-$WEND A"
echo "=================================================================="
$MPIRUN -n $N $MPIFLAGS $PYTHON $BASE/spectro_grid_synthe_mpi.py \
    --grid-dir "$MODELS" --specdir "$H5DIR" \
    --wstart $WSTART --wend $WEND --resolu $RESOLU

echo "=================================================================="
echo " STAGE C — assemble ATLAS o photometry grid"
echo "=================================================================="
$PYTHON $BASE/assemble_grid.py \
    --specdir "$H5DIR" \
    --filter "$BASE/filters/Misc_Atlas.o.txt" \
    --filter-name o --filter-desc "ATLAS o" \
    --ext 0.846 --out "$OUTGRID"

echo "DONE -> $OUTGRID"
