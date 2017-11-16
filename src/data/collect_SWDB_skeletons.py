#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 09:52:44 2017

@author: ajaver
"""
import pymysql
import pandas as pd
from collect_CeNDR_skeletons import collect_skeletons, add_sets_index

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/work-in-progress/work_in_progress/new_features')

if __name__ == '__main__':
    main_file = '/Users/ajaver/Desktop/SWDB_skel_smoothed_v2.hdf5'
    
    conn = pymysql.connect(host='localhost', database='single_worm_db')
    
    sql = '''
    SELECT *, 
    CONCAT(results_dir, '/', base_name, '_skeletons.hdf5') AS skel_file,
    n_valid_skeletons/(total_time*fps) AS frac_valid
    FROM experiments_full AS e
    JOIN results_summary ON e.id = experiment_id
    WHERE total_time < 905
    AND total_time > 295
    AND strain != '-N/A-'
    AND exit_flag = 'END'
    AND n_valid_skeletons > 120*fps
    ORDER BY frac_valid
    '''
    experiments_df = pd.read_sql(sql, con=conn)
    #filter strains that have at least 10 videos
    experiments_df = experiments_df.groupby('strain').filter(lambda x: len(x) > 10)
    experiments_df = experiments_df.rename(columns={'results_dir':'directory'})
    
    valid_cols = ['id', 'base_name', 'directory', 'strain', 'fps']
    experiments_df = experiments_df[valid_cols]
    collect_skeletons(experiments_df,  main_file)
    add_sets_index(main_file, val_frac = 0.1, test_frac = 0.1)
