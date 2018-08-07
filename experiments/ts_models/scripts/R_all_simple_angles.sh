#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q@compG004 -l gputype=p100
#$ -l gpu=1 -pe shmem 1

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0
source activate pytorch-v0.4.0-cuda8.0-venv 

echo "Username: " `whoami`
echo $HOME
echo cuda_id: $CUDA_VISIBLE_DEVICES

python $HOME/Github/classify_strains/experiments/ts_models/train.py \
--model_name 'simple' --set_type 'angles' --n_epochs 1000 --batch_size 8 \
--num_workers 1 --optimizer 'sgd' --lr 0.0001 \
 --copy_tmp '/tmp/avelino'$CUDA_VISIBLE_DEVICES \
--init_model_path 'logs/angles_20180524_173345_simple_lr0.0001_batch8/model_best.pth.tar'

echo "Finished at :"`date`
exit 0

