
#!/bin/sh
#PBS -l walltime=24:00:00
## This tells the batch manager to limit the walltime for the job to XX hours, YY minutes and ZZ seconds.

#PBS -l select=1:ncpus=2:mem=16gb:ngpus=1
## This tells the batch manager to use NN node with MM cpus and PP gb of memory per node with QQ gpus available.

#PBS -q gpgpu
## This tells the batch manager to enqueue the job in the general gpgpu queue.

module load anaconda3
module load cuda
## This job requires CUDA support.
source activate tierpsy

## copy temporary files
cp $WORK/classify_strains/train_set/SWDB_skel_smoothed.hdf5 $TMPDIR/

python $HOME/classify_strains/nn/train.py --model_name resnet18 --dataset SWDB --data_file None --is_reduced True --sample_size_seconds 10 --sample_frequency_s 0.04 --n_batch 32 --transform_type angles --is_normalized False --n_epochs 200
## This tells the batch manager to execute the program cudaexecutable in the cuda directory of the users home directory.
