
#!/bin/sh
#PBS -l walltime=24:00:00
## This tells the batch manager to limit the walltime for the job to XX hours, YY minutes and ZZ seconds.

#PBS -l select=1:ncpus=2:mem=16gb:ngpus=1
## This tells the batch manager to use NN node with MM cpus and PP gb of memory per node with QQ gpus available.

module load anaconda3
module load cuda
## This job requires CUDA support.
source activate tierpsy

## copy temporary files
cp $WORK/classify_strains/trained_models/ae_w_embeddings/*_checkpoint.pth.tar $TMPDIR
mkdir -p $TMPDIR/ae_w_embeddings
cp $WORK/classify_strains/train_set/CeNDR_skel_smoothed.hdf5 $TMPDIR/ae_w_embeddings

python $HOME/classify_strains/experiments/skeletons_autoencoder/get_embeddings.py

cp $TMPDIR/ae_w_embeddings/*_embeddings.hdf5 $WORK/classify_strains/trained_models/ae_w_embeddings/