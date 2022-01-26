#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 19:26:36 2019

@author: panyunbei
"""


import numpy as np
import matplotlib.pyplot as plt

m = []
m.append(open('./data_for_modelling/appros4a1.txt'))
m.append(open('./data_for_modelling/appros4a2.txt'))
m.append(open('./data_for_modelling/appros4a3.txt'))
m.append(open('./data_for_modelling/appros4a4.txt'))

s = []
for i in m:
    s.append(i.readlines())

data = []
for i in s:
    data.append([elem.strip().split('\t') for elem in i])
    
w = np.arange(0.001, 1.001, 0.001)

error = []
for i in data:
	error_i = []
	m = i[2:]
	for j in m:
		error_i.append(float(j[0]))
	error.append(error_i)

print(error[0][3])
print(error[1][3])
########################################################################################
####################################### Plot ###########################################    

font1={'weight':'medium', 'size':21}

fig,ax0 = plt.subplots(figsize=(6,6))

ax0.plot(w, error[0], c='red', label='$\lambda=10^{-4}$, $\\bar{\\alpha}=2*10^{-4}$')
ax0.plot(w, error[1], c='blue',label='$\lambda=10^{-4}$, $\\bar{\\alpha}=10^{-4}$')
ax0.plot(w, error[2], c='green',label='$\lambda=10^{-4}$, $\\bar{\\alpha}=6*10^{-5}$')
ax0.plot(w, error[3], c='gold',label='$\lambda=10^{-4}$, $\\bar{\\alpha}=2*10^{-5}$')

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
