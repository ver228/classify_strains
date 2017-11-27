#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 18:28:35 2017

@author: ajaver
"""
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from flow import ROIFlowBase, shift_and_normalize
from models import AE3D
if __name__ == '__main__':
    mask_file = '/Users/ajaver/OneDrive - Imperial College London/aggregation/N2_1_Ch1_29062017_182108_comp3.hdf5'
    feat_file = mask_file.replace('.hdf5', '_featuresN.hdf5')
        
    g = ROIFlowBase(mask_file, feat_file)
    
    model = AE3D()
    #%%
    snippet_size = 128
    
    rois_collected = {}
    for frame_number in range(snippet_size):
        frame_rois = g._get_frame_rois(frame_number)
        for w, (roi, _) in frame_rois.items():
            if not w in rois_collected:
                rois_collected[w] = []
            rois_collected[w].append(roi[None, ...])
    
    all_snippets = []
    for w, rois_l in rois_collected.items():
        if len(rois_l) < snippet_size:
            continue
        
        snippet = np.concatenate(rois_l[:snippet_size])
        snippet = shift_and_normalize(snippet.astype(np.float32)) + 0.5
        
        all_snippets.append(snippet[None, None, ...])
    #%%
    batch_size = 4
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    import tqdm
    for epoch in range(1000):
        pbar = tqdm.tqdm(range(0, len(all_snippets), batch_size))
        for ii in pbar:
            snippets = all_snippets[ii:ii+batch_size]
            S = np.concatenate(snippets)
            S = torch.from_numpy(S)
            S = Variable(S)
            
            output = model(S)
            loss = criterion(output, S)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description('Epoch {} loss {}'.format(epoch, loss.data[0]), refresh=False)
    #%%
    import matplotlib.pylab as plt
    import matplotlib.animation as animation
    import numpy as np
    
    
    
    dpi = 100
    def ani_frame(n_batch):
        
        ori = S.data.squeeze().numpy()[n_batch]
        dat = output.data.squeeze().numpy()[n_batch]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        dd = np.hstack((ori[0], dat[0]))
        im = ax.imshow(dd,cmap='gray',interpolation='nearest')
        im.set_clim([0,1])
        fig.set_size_inches([8,4])
        plt.tight_layout()
        
    
        
    
    
        def update_img(n):
            tmp = np.hstack((ori[n], dat[n]))
            im.set_data(tmp)
            return im
    
        #legend(loc=0)
        ani = animation.FuncAnimation(fig,update_img, 128, interval = 40)
        writer = animation.writers['ffmpeg'](fps = 25)
    
        ani.save('demo{}.mp4'.format(n_batch),writer=writer,dpi=dpi)
        return ani
    
    for kk in range(S.size(0)):
        ani = ani_frame(kk)
        plt.show()
    #%%
#    plt.figure()
#    im=plt.imshow(dat[0], interpolation='none', cmap='gray')
#    for row in dat:
#        im.set_data(row)
#        plt.pause(0.02)
#    plt.show()
#    