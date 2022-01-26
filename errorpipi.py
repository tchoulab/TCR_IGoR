#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:59:20 2019

@author: panyunbei
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:31:06 2019

@author: panyunbei
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import log,exp

m = open('./data_for_modelling/res2.txt')
s = m.readlines()

Z = [elem.strip().split('\t') for elem in s]
Z = Z[1:]
#print(complex(Z[98][0]))
#if 'I' in Z[98][1]:
#    print('True')

for i in range(len(Z)):
    for j in range(len(Z[1])):
            Z[i][j] = float(Z[i][j])


font1={'weight':'medium', 'size':21}

fig,ax0 = plt.subplots(figsize=(6,6))
#fig = plt.figure()
#ax0 = Axes3D(fig)

alpha_bar = np.arange(-5.5, -2.475, 0.025)
sigma = np.arange(-3.5, -0.965, 0.035)

#error = Z
#for i in range(len(Z)):
#    for j in range(len(Z[0])):
#        error[i][j] = exp(Z[i][j])+0.001

#error = np.array(error)
si = [10**i for i in sigma]
si = np.array(si)

#ax0.plot_surface(alpha_bar, si, error, rstride = 1, cstride = 1, cmap = plt.cm.hot)
#plt.colorbar(cset)
plt.tick_params(labelsize=18)



ax0.pcolormesh(alpha_bar, si, Z)

#x = [0, 1, 2, 3, 4, 5, 6]
x = [-5.5,-6+log(4,10),-6+log(5,10),-6+log(6,10),-6+log(7,10),-6+log(8,10),
     -6+log(9,10),-5,-5+log(2,10),-5+log(3,10),-5+log(4,10),-5+log(5,10),-5+log(6,10),
     -5+log(7,10),-5+log(8,10),-5+log(9,10), -4, -4+log(2,10),-4+log(3,10),
     -4+log(4,10),-4+log(5,10),-4+log(6,10),-4+log(7,10),-4+log(8,10),-4+log(9,10),
     -3,-3+log(2,10), -3+log(3,10), -2.5]
#y = [0.000065, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]

#x = [-5,-5+log(2,10),-5+log(3,10),-5+log(4,10),-5+log(5,10),-5+log(6,10),
#     -5+log(7,10),-5+log(8,10),-5+log(9,10), -4, -4+log(2,10),-4+log(3,10),
#     -4+log(4,10),-4+log(5,10),-4+log(6,10),-4+log(7,10),-4+log(8,10),-4+log(9,10),
#     -3]
#plt.xticks(x, ('$10^{-5}$','','','','','','','','','$10^{-4}$','','','','','','','','','$10^{-3}$'))
plt.xticks(x, ('','','','','','','','$10^{-5}$','','','','','','','','','$10^{-4}$',
               '','','','','','','','','$10^{-3}$','','',''))
#plt.yticks(y, ('0.0', '', '0.002', '','0.004', '','0.006','', '0.008','','0.010'))

plt.show()