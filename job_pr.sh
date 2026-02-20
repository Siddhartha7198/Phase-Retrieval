#!/bin/bash

################
#
# Setting slurm options
#
################

# lines starting with "#SBATCH" define your jobs parameters


#SBATCH --partition graphic

# telling slurm how many instances of this job to spawn (typically 1)
#SBATCH --ntasks 1

# setting number of CPUs per task (1 for serial jobs)
#SBATCH --cpus-per-task 1

# setting memory requirements
#SBATCH --mem-per-cpu 50G

# propagating max time for job to run - choose zone of the formats below
#SBATCH --time 2-00:00:00

#SBATCH --array=1-25

# Setting the name for the job
#SBATCH --job-name newcrop

# setting notifications for job
# accepted values are ALL, BEGIN, END, FAIL, REQUEUE
#SBATCH --mail-type FAIL

# telling slurm where to write output and error
# this will create the output files in the current directory should you wish
#for them to be put elsewhere use absolute paths e.g. /home/<user>/queue/output
#SBATCH --output=/data/finite/poddar/phase/%A_%a.out
#SBATCH --error=/data/finite/poddar/phase/%A_%a.out

################
#
# copying your data to /scratch
#
################

# create local folder on ComputeNode
# ALWAYS copy any relevant data for your job to local disk to speed up your job
# and decrease load on the fileserver
scratch=/scratch/$USER/$SLURM_JOB_ID
mkdir -p $scratch
cp /data/finite/poddar/phase/Phase_3d_opt.py $scratch/PhaseRetrieval.py
cp /data/finite/poddar/phase/pr_3d.py $scratch
cp /data/finite/poddar/gan/diffgan/3d_den_ribosome_new.npy $scratch/3d_den.npy
cd $scratch

# dont access /home after this line

# if needed load modules here
module purge
module load python/3.8.3


# if needed add export variables here
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


################
#
# run the program
#
################
srun python3 pr_3d.py $SLURM_ARRAY_TASK_ID

# copy results to data
mkdir -p /data/finite/poddar/phase/ribosome/$SLURM_JOB_NAME/
cp pr_recon_$SLURM_ARRAY_TASK_ID.npy /data/finite/poddar/phase/ribosome/$SLURM_JOB_NAME/
cp pr_error_$SLURM_ARRAY_TASK_ID.npy /data/finite/poddar/phase/ribosome/$SLURM_JOB_NAME/
cp pr_recon_$SLURM_ARRAY_TASK_ID.mat /data/finite/poddar/phase/ribosome/$SLURM_JOB_NAME/

cd

# clean up scratch
rm -rf $scratch
unset scratch

exit 0
