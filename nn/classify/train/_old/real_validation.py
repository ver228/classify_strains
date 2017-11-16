#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:15:26 2017

@author: ajaver
"""
import sys
if sys.platform == 'linux':
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import tables
import os
from keras.models import load_model
import itertools

from sklearn.metrics import confusion_matrix
from collections import Counter
from skeletons_flow import SkeletonsFlow, _h_angles, SWDB_WILD_ISOLATES, CeNDR_DIVERGENT_SET

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    #based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cmap=plt.cm.Blues
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    np.set_printoptions(precision=2)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%1.2f' % cm[i, j],
                 horizontalalignment="center",
                 fontsize =12,
                 color="white" if cm[i, j] > thresh else "black")
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def _h_show_results(all_results, strains_codes, save_name):
    # group results for all the strains
    y_vec_dict = {}
    for row, y_l in all_results:
        strain_id = row['strain']
        
        y = np.sum(y_l, axis=0)
        if not strain_id in y_vec_dict:
            y_vec_dict[strain_id]  = y
        else:
             y_vec_dict[strain_id] += y
    
    # print the top 5 predictions per strain
    for strain, vec in y_vec_dict.items():
        prob = vec/np.sum(vec)
        ind = np.argsort(prob)[:-6:-1]
        s_sort = strains_codes.loc[ind, 'strain']
        
        print(strain)
        for s, p in zip(s_sort, prob[ind]):
            print('{} - {:.2f}'.format(s,p*100))
        print('*************')
    
    # collect the best results per experiment/video
    y_pred_dict = {}
    for row, y_l in all_results:
        yys = np.argmax(y_l, axis=1)
        
        exp_id = row['experiment_id']
        if not exp_id in y_pred_dict:
            y_pred_dict[exp_id]  = []
        
        y_pred_dict[exp_id] += list(yys)
    
    # #%% #collect sorted predictions
    # y_pred_all = {}
    # for row, y_l in all_results:
    #     yys = np.argsort(y_l, axis=1)
        
    #     exp_id = row['experiment_id']
    #     if not exp_id in y_pred_all:
    #         y_pred_all[exp_id]  = []
        
    #     y_pred_all[exp_id] += [np.argsort(y_l)[:, ::-1]]
    
    # for k, v in y_pred_all.items():
    #     y_pred_all[k] = np.vstack(v)
    
    
    # get the accuracy percentage per chunk
    df, _ = zip(*all_results)
    df = pd.concat(df, axis=1).T
    
    y_true, y_pred = [], []
    
    chuck_p = []
    for _, row in df.drop_duplicates('experiment_id').iterrows():
        y_true.append(row['strain'])
        
        y_l = y_pred_dict[row['experiment_id']]
        
        dd = Counter(y_l).most_common(1)[0][0]
        y_pred.append(strains_codes.loc[dd]['strain']) 
        
        chuck_p += [(row['strain'], x) for x in strains_codes.loc[y_l]['strain'].values]
    
    #print results
    labels = sorted(list(set(y_true)))
    dd = sum(x[0] == x[1] for x in chuck_p)
    print('Accuracy by chunk: {}'.format(dd/len(chuck_p)))
    
    dd = sum(x[0] == x[1] for x in zip(y_pred, y_true))
    print('Accuracy by video: {}'.format(dd/len(y_true)))
    
    
    
    
    with PdfPages(save_name) as pdf:
        cm_c_chunk = confusion_matrix(*zip(*chuck_p), labels=labels)
        plt.figure(figsize=(21,21))
        plot_confusion_matrix(cm_c_chunk, 
                              labels,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues,
                              normalize = True
                              )
        plt.title('Confusion matrix by Chunk')
        pdf.savefig()
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(21,21))
        plot_confusion_matrix(cm, 
                              labels,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues,
                              normalize = True
                              )
        plt.title('Confusion matrix by Video')
        pdf.savefig()


def get_model_results(model_path,
                    data_file,
                    is_reduced = False,
                    is_angle = False,
                    set_type = 'test'):
    
    valid_strains = None
    if is_reduced:
        if 'SWDB' in data_file:
            valid_strains = SWDB_WILD_ISOLATES
        elif 'CeNDR' in data_file: 
            valid_strains = CeNDR_DIVERGENT_SET

    print(valid_strains)
    print('loading model...')
    model = load_model(model_path)
     
    print('loading data...')    
    gen = SkeletonsFlow(data_file = data_file, 
                       n_batch = 32, 
                       set_type = set_type,
                       valid_strains = valid_strains,
                       is_angle = is_angle
                       )

    all_results = []
    for ii, (irow, row) in enumerate(gen.skeletons_ranges.iterrows()):
        print(ii+1, len(gen.skeletons_ranges))
        fps = row['fps']
        
        #get the expected row indexes
        row_indices_r = np.linspace(0, gen.sample_size_frames_s, gen.n_samples)*fps
        row_skels = []
        sample_size_frames = int(gen.sample_size_frames_s*fps)
        for ini_r in range(row['ini'], row['fin'], int(sample_size_frames/2)):
            fin_r = ini_r + sample_size_frames
            if fin_r > row['fin']:
                continue 
            
            row_indices = row_indices_r + ini_r
            row_indices = np.round(row_indices).astype(np.int32)
            
            with tables.File(gen.data_file, 'r') as fid:
                skeletons = fid.get_node('/skeletons_data')[row_indices, :, :]
            body_coords = np.mean(skeletons[:, gen.body_range[0]:gen.body_range[1]+1, :], axis=1)
            
            if not is_angle:
                skeletons -= body_coords[:, None, :]
                row_skels.append(skeletons)
            else:
                X, _ = _h_angles(skeletons)
                X = X[..., None]
                row_skels.append(X)
                
        batch_data = np.array(row_skels)
        Y = model.predict(batch_data)
        
        all_results.append((row, Y))
    

    with pd.HDFStore(gen.data_file, 'r') as fid:
        strains_codes = fid['/strains_codes']
    
    # print the confusion matrix per chuck and video
    save_name = os.path.splitext(model_path)[0]
    save_name += '_cm.pdf'
    
    _h_show_results(all_results, strains_codes, save_name)


if __name__ == '__main__':
  import fire
  fire.Fire(get_model_results)
