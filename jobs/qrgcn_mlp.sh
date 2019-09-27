#!/bin/bash
#SBATCH --job-name=qrgcn_mlp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

PROJ_FOLDER=graphqembed
LOG_FOLDER=output

# Copy data to scratch
cp -r $HOME/$PROJ_FOLDER $TMPDIR
cd $TMPDIR/$PROJ_FOLDER

# Run experiment
source activate pygeom
srun python -u -m netquery.bio.train \
--model="qrgcn" \
--log_dir=$LOG_FOLDER \
--model_dir=$LOG_FOLDER \
--cuda \
--lr=0.01 \
--readout="mlp" \
--dropout=0  \
--weight_decay=0.0

cp -r $TMPDIR/$PROJ_FOLDER/$LOG_FOLDER $HOME/$PROJ_FOLDER
