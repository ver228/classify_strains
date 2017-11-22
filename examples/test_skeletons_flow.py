#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:46:12 2017

@author: ajaver
"""

def test_transforms(data_file):
    for tt in ['angles', 'eigenworms', 'eigenworms_full', 'xy']:
        for is_n in [False, True]:
            print(tt)
            gen = SkeletonsFlowShuffled(n_batch = 2, 
                          data_file = data_file, 
                          transform_type = tt,
                          is_normalized = is_n,
                          set_type = 'train', 
                          sample_size_seconds = 10, 
                          sample_frequency_s=1/25.
                          )
            X,Y = next(gen)
            print(X.shape)
            
           
    
if __name__ == '__main__':
    import os
    import sys
    
    src_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    sys.path.append(src_dir)
    
    from classify.flow import SkeletonsFlowFull, SkeletonsFlowShuffled, get_datset_file
    
    dataset = 'CeNDR'
    #dataset = 'SWDB'
    
    data_file = get_datset_file(dataset)
    data_file = data_file.replace('/CeNDR/', '/_old/')
    
    #test_transforms(data_file)
    
    
    gen = SkeletonsFlowFull(
                            n_batch = 32, 
                          data_file = data_file,
                          set_type = 'test', 
                          sample_size_seconds = 10, 
                          sample_frequency_s=1/25.,
                          is_torch = True
                          )
    print(len(gen))
    for ii, (X,Y) in enumerate(gen):
        print(ii)
        #print(ii, X.size(), Y.size())
        