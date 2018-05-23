#!/bin/bash

#$ -P rittscher.prjb -q short.qb


echo "Username: " `whoami`
source $HOME/ini_session.sh
python /users/rittscher/avelino/Github/classify_strains/experiments/ts_models/collect_emb.py
exit 0
