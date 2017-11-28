#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 00:24:15 2017

@author: ajaver
"""

if __name__ == '__main__':
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