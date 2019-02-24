import numpy as np
import torch


def fastdist(x, y=None, verbose=False):
    '''
    Fast pytorch-cuda computation of pairwise distance matrix, largely inspired by discussion at :
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
    
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
        
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag())
        
    # Check for negative numbers 
    if verbose :
        minimum = dist.min()   
        if minimum < 0 :    
            print('Min. neg. dist. squared : ' + str(minimum) + 
                  ' relative error =  ' + str( -minimum/dist.median()) )
    # clamp
    dist = torch.clamp(dist, 0.0, np.inf)
    
    return np.sqrt(dist)