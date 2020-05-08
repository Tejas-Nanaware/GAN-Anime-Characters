#!/bin/sh
#SBATCH -A iit111
#SBATCH --job-name="gan_anime"
#SBATCH --output="keras.%j.%N.out"
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:4
#SBATCH -t 02:00:00
#Run the cluster
module load singularity # load the singularity module
singularity exec --nv /share/apps/gpu/singularity/images/keras/keras-v2.2.4-tensorflow-v1.12-gpu-20190214.simg python3 DCGAN.py
