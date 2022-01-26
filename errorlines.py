#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 19:26:36 2019

@author: panyunbei
"""


import numpy as np
import matplotlib.pyplot as plt

m = []
m.append(open('./data_for_modelling/errors2.txt'))

s = []
for i in m:
    s.append(i.readlines())

data = []
for i in s:
    data.append([elem.strip().split('\t') for elem in i])
#print(data)
    
datai = data[0][1:]


#w = np.arange(0.01, 1.01, 0.01)

w = np.arange(0.005, 1.005, 0.005)

error_0 = []
error_1 = []
error_2 = []
error_3 = []
for i in datai:
    error_0.append(float(i[0]))
    error_1.append(float(i[1]))
    error_2.append(float(i[2]))
    error_3.append(float(i[3]))
 
########################################################################################
####################################### Plot ###########################################    

font1={'weight':'medium', 'size':21}

fig,ax0 = plt.subplots(figsize=(6,6))

ax0.plot(w, error_0, c='b', label='$\\bar{\\alpha}=2*10^{-4}$, $\lambda=10^{-2}$')
ax0.plot(w, error_1, c='g', label='$\\bar{\\alpha}=4*10^{-4}$, $\lambda=10^{-2}$')
ax0.plot(w, error_2, c='gold', label='$\\bar{\\alpha}=6*10^{-4}$, $\lambda=10^{-2}$')
ax0.plot(w, error_3, c='red', label='$\\bar{\\alpha}=8*10^{-4}$, $\lambda=10^{-2}$')
#    ax0.plot(alpha[i-1:i+1], sigma[4][i-1:i+1], c=(0, t4[i-1], 0))

#x = [-6,-6+log(2,10),-6+log(2,10),-6+log(4,10),-6+log(5,10),-6+log(6,10),-6+log(7,10),-6+log(8,10),
#     -6+log(9,10),-5,-5+log(2,10),-5+log(3,10),-5+log(4,10),-5+log(5,10),-5+log(6,10),-5+log(7,10),
#     -5+log(8,10),-5+log(9,10),-4,-4+log(2,10),-4+log(3,10),-4+log(4,10),-4+log(5,10),-4+log(6,10),
#     -4+log(7,10),-4+log(8,10),-4+log(9,10),-3,-3+log(2,10),-3+log(3,10),-3+log(4,10),-3+log(5,10),
#     -3+log(6,10),-3+log(7,10),-3+log(8,10),-3+log(9,10),-2,]

#plt.xticks(x, ('$10^{-6}$','','','','','','','','','$10^{-5}$','','','','','','','','','$10^{-4}$','','','','','','','',
#               '','$10^{-3}$','','','','','','','','','$10^{-2}$'))
#

ax0.set_yscale('log')

plt.tick_params(labelsize=18)
plt.legend()
plt.show()
