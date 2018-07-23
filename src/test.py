# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:49:11 2016

@author: Srijita
"""
import numpy as np
sri=np.array([[1,0,3,2],[4,5,0,2],[7,0,9,2],[1,2,3,4],[3,4,5,6],[6,7,8,9]])
sri1 = sum(map(np.array, sri))
print sri1
def average(x):
    return x/2
sri2=map(average,sri1)
print sri2
sri1=[[1],[2],[3],[4]]
print np.mean(sri1,axis=0)

