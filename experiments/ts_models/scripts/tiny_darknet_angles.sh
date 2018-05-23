#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -l gputype=p100
#$ -l gpu=1 -pe shmem 1

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0
source activate pytorch-v0.4.0-cuda8.0-venv 

echo "Username: " `whoami`
echo $HOME
echo cuda_id: $CUDA_VISIBLE_DEVICES

python $HOME/Github/classify_strains/experiments/ts_models/train.py --is_tiny --model_name 'darknet' --set_type 'angles' --n_epochs 1000 --batch_size 8 --num_workers 1 --lr 0.0001  --copy_tmp '/tmp/avelino'

echo "Finished at :"`date`
exit 0

