#!/bin/bash
#SBATCH --job-name=qrgcn_sum
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --time=30:00:00
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
srun python -u -m netquery.bio.train_rgcn \
--log_dir=$LOG_FOLDER \
--model_dir=$LOG_FOLDER \
--cuda \
--lr=0.01 \
--readout="sum" \
--dropout=0  \
--weight_decay=0.0 \
--num_passes=3

cp -r $TMPDIR/$PROJ_FOLDER/$LOG_FOLDER $HOME/$PROJ_FOLDER
