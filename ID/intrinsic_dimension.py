#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:19:48 2018

@author: alessioansuini@gmail.com
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn import linear_model
from math import sqrt


def estimate(X,fraction=0.9,verbose=True):
    
    '''
        Estimates the intrinsic dimension of a system of points from
        the matrix of their distances X
        
        Args:
        X : 2-D Matrix X (n,n) where n is the number of points
        fraction : fraction of the data considered for the dimensionality
        estimation (default : fraction = 0.9)

        Returns:            
        x : log(mu)    (*)
        y : -(1-F(mu)) (*)
        reg : linear regression y ~ x structure obtained with scipy.stats.linregress
        (reg.slope is the intrinsic dimension estimate)
        r : determination coefficient of y ~ x
        pval : p-value of y ~ x
            
        (*) See cited paper for description
        
        Usage:
            
        _,_,reg,r,pval = estimate(X,fraction=0.85)
            
        The technique is described in : 
            
        "Estimating the intrinsic dimension of datasets by a 
        minimal neighborhood information"       
        Authors : Elena Facco, Maria d’Errico, Alex Rodriguez & Alessandro Laio        
        Scientific Reports 7, Article number: 12140 (2017)
        doi:10.1038/s41598-017-11873-y
    
    ''' 
    
    Y = np.sort(X,axis=1,kind='quicksort')
    # clean first neighbour zero distance values
    k1 = Y[:,1]
    k2 = Y[:,2]
    good = k1 > 1e-8
    k1 = k1[good]
    k2 = k2[good]
    
    if verbose:
        print('Fraction good = ' + str( k1.shape[0]/Y.shape[0] ) )
   
    mu = np.sort(np.divide(k2, k1), axis=None,kind='quicksort')    
    npoints = int(np.floor(good.shape[0]*fraction))
    y = np.ones((1,good.shape[0]-1)) - np.arange(good.shape[0]-1,dtype= np.float64) / good.shape[0]
    y = np.squeeze(y)
    y = -np.log(y)
    x = np.log(mu[0:good.shape[0]-1])
    del Y
        
    #reg = linregress(x[0:npoints], y[0:npoints])
    regr = linear_model.LinearRegression()
    regr.fit(x[0:npoints,np.newaxis],y[0:npoints,np.newaxis])    
    r,pval = pearsonr(x[0:npoints], y[0:npoints])               
    return (x,y,regr.coef_[0][0],r,pval)
  

def block_analysis(X, blocks=list(range(1, 21)), fraction=0.9):
    
    '''
        Perform a block-analysis of a system of points from
        the matrix of their distances X
        
        Args:
        X : 2-D Matrix X (n,n) where n is the number of points
        blocks : blocks specification, is a list of integers from
        1 to N_blocks where N_blocks is the number of blocks (default : N_blocks = 20)
        fraction : fraction of the data considered for the dimensionality
        estimation (default : fraction = 0.9)

    
    '''
    

    n = X.shape[0]
    dim = np.zeros(len(blocks))
    std = np.zeros(len(blocks))
    n_points = []
   
    for b in blocks:        
        # split indexes array
        idx = np.random.permutation(n)
        npoints = int(np.floor((n / b )))
        idx = idx[0:npoints*b]
        split = np.split(idx,b)      
        tdim = np.zeros(b)
        for i in range(b):            
            I = np.meshgrid(split[i], split[i], indexing='ij')
            tX = X[I]
            _,_,reg,_,_ = estimate(tX,fraction=fraction,verbose=False)
            tdim[i] = reg          
        dim[blocks.index(b)] = np.mean(tdim)
        std[blocks.index(b)] = np.std(tdim)
        n_points.append(npoints)
    return dim,std,n_points