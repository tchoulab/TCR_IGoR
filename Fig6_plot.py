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
from math import log,exp
import matplotlib

############################
m1 = open('errordeltadeltaeta4.txt')
s1 = m1.readlines()

Z1 = [elem.strip().split('\t') for elem in s1]
Z1 = Z1[1:]

for i in range(len(Z1)):
    for j in range(len(Z1[1])):
        Z1[i][j] = float(Z1[i][j])

#print(len(Z1))
#print(len(Z1[1]))
#print(np.min(Z1))
#print(np.max(Z1))

########
m2 = open('errordeltadeltaeta5.txt')
s2 = m2.readlines()

Z2 = [elem.strip().split('\t') for elem in s2]
Z2 = Z2[1:]

for i in range(len(Z2)):
    for j in range(len(Z2[1])):
        Z2[i][j] = float(Z2[i][j])

#print(len(Z2))
#print(len(Z2[1]))
#print(np.min(Z2))
#print(np.max(Z1), np.max(Z2))

########
m3 = open('errordeltadeltaeta6.txt')
s3 = m3.readlines()

Z3 = [elem.strip().split('\t') for elem in s3]
Z3 = Z3[1:]

for i in range(len(Z3)):
    for j in range(len(Z3[1])):
        Z3[i][j] = float(Z3[i][j])

#print(len(Z3))
#print(len(Z3[1]))
#print(np.min(Z3))
#print(np.max(Z3))

############################
m4 = open('errordeltapieta4.txt')
s4 = m4.readlines()

Z4 = [elem.strip().split('\t') for elem in s4]
Z4 = Z4[1:]

for i in range(len(Z4)):
    for j in range(len(Z4[1])):
        Z4[i][j] = float(Z4[i][j])

#print(len(Z4))
#print(len(Z4[1]))
#print(np.min(Z4))
#print(np.max(Z4))

########
m5 = open('errordeltapieta5.txt')
s5 = m5.readlines()

Z5 = [elem.strip().split('\t') for elem in s5]
Z5 = Z5[1:]

for i in range(len(Z5)):
    for j in range(len(Z5[1])):
        Z5[i][j] = float(Z5[i][j])

#print(len(Z5))
#print(len(Z5[1]))
#print(np.min(Z5))
#print(np.max(Z5), np.max(Z5))

########
m6 = open('errordeltapieta6.txt')
s6 = m6.readlines()

Z6 = [elem.strip().split('\t') for elem in s6]
Z6 = Z6[1:]

for i in range(len(Z6)):
    for j in range(len(Z6[1])):
        Z6[i][j] = float(Z6[i][j])

#print(len(Z6))
#print(len(Z6[1]))
#print(np.min(Z6))
#print(np.max(Z6))
########################################

font1={'weight':'medium', 'size':21}

#from matplotlib import rcParams
#config = {
    #"text.usetex":True
    #"font.family":'italic',
    #"mathtext.fontset":'stix'
#    }
#rcParams.update(config)
fig = plt.figure(figsize=(11,6))
#fig,axs = plt.subplots(1,3)

error1 = Z1
error2 = Z2
error3 = Z3
error4 = Z4
error5 = Z5
error6 = Z6
alpha_bar1 = np.arange(-5.5, -2.475, 0.025)
alpha_bar2 = np.arange(-5.5, -2.475, 0.05)
sigma1 = np.arange(0.05, 8.05, 0.04)
sigma2 = np.arange(-7.5, -1.975, 0.025)
si = [10**i for i in sigma2]
si = np.array(si)
#for i in range(len(Z1)):
#    for j in range(len(Z1[0])):
#        error1[i][j] = log(exp(Z1[i][j])+0.001)
#for i in range(len(Z2)):
#    for j in range(len(Z2[0])):
#        error2[i][j] = log(exp(Z2[i][j])+0.001)
#for i in range(len(Z3)):
#    for j in range(len(Z3[0])):
#        error3[i][j] = log(exp(Z3[i][j])+0.001)

for i in range(len(Z1)):
    for j in range(len(Z1[0])):
        error1[i][j] = log((exp(Z1[i][j])+0.001),10)
for i in range(len(Z2)):
    for j in range(len(Z2[0])):
        error2[i][j] = log((exp(Z2[i][j])+0.001),10)
for i in range(len(Z3)):
    for j in range(len(Z3[0])):
        error3[i][j] = log((exp(Z3[i][j])+0.001),10)
#print(round(np.min(error1)))
#print(np.min(error2))
#print(np.min(error3))

norm1 = matplotlib.colors.Normalize(vmin=-3, vmax=0)
norm2 = matplotlib.colors.Normalize(vmin=-6, vmax=0)


ax1 = plt.subplot(2,3,1)
H1 = plt.pcolormesh(alpha_bar2, sigma1, error1, norm = norm1, rasterized=True)

ax2 = plt.subplot(2,3,2)
H2 = plt.pcolormesh(alpha_bar1, sigma1, error2, norm = norm1, rasterized=True)

ax3 = plt.subplot(2,3,3)
H3 = plt.pcolormesh(alpha_bar1, sigma1, error3, norm = norm1, rasterized=True)

ax4 = plt.subplot(2,3,4)
H4 = plt.pcolormesh(alpha_bar2, si, error4, norm = norm2, rasterized=True)

ax5 = plt.subplot(2,3,5)
H5 = plt.pcolormesh(alpha_bar1, si, error5, norm = norm2, rasterized=True)

ax6 = plt.subplot(2,3,6)
H6 = plt.pcolormesh(alpha_bar1, si, error6, norm = norm2, rasterized=True)

fig.subplots_adjust(right=0.9)
l = 0.92
b1 = 0.55
b2 = 0.12
w = 0.015
h1 = 1 - 1.22*b1
h2 = 1 - 5.5*b2

rect1 = [l,b1,w,h1]
rect2 = [l,b2,w,h2]
cbar_ax1 = fig.add_axes(rect1)
cbar_ax2 = fig.add_axes(rect2)
cb1 = plt.colorbar(H3, cax=cbar_ax1, shrink=0.5)
cb2 = plt.colorbar(H6, cax=cbar_ax2, shrink=0.5)

x = [-5.5,-6+log(4,10),-6+log(5,10),-6+log(6,10),-6+log(7,10),-6+log(8,10),
     -6+log(9,10),-5,-5+log(2,10),-5+log(3,10),-5+log(4,10),-5+log(5,10),-5+log(6,10),
     -5+log(7,10),-5+log(8,10),-5+log(9,10), -4, -4+log(2,10),-4+log(3,10),
     -4+log(4,10),-4+log(5,10),-4+log(6,10),-4+log(7,10),-4+log(8,10),-4+log(9,10),
     -3,-3+log(2,10), -3+log(3,10), -2.5]

#y = [0.000065, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
ax1.set_xticks(x)
ax1.set_xticklabels(['','','','','','','','$10^{-5}$','','','','','','','','','$10^{-4}$',
              '','','','','','','','','$10^{-3}$','','',''])
#plt.yticks(y, ('0.0', '', '0.002', '','0.004', '','0.006','', '0.008','','0.010'))
ax2.set_xticks(x)
ax2.set_xticklabels(['','','','','','','','$10^{-5}$','','','','','','','','','$10^{-4}$',
              '','','','','','','','','$10^{-3}$','','',''])
ax3.set_xticks(x) 
ax3.set_xticklabels(['','','','','','','','$10^{-5}$','','','','','','','','','$10^{-4}$',
              '','','','','','','','','$10^{-3}$','','',''])
ax4.set_xticks(x)
ax4.set_xticklabels(['','','','','','','','$10^{-5}$','','','','','','','','','$10^{-4}$',
              '','','','','','','','','$10^{-3}$','','',''])
ax5.set_xticks(x)
ax5.set_xticklabels(['','','','','','','','$10^{-5}$','','','','','','','','','$10^{-4}$',
              '','','','','','','','','$10^{-3}$','','',''])
ax6.set_xticks(x)
ax6.set_xticklabels(['','','','','','','','$10^{-5}$','','','','','','','','','$10^{-4}$',
              '','','','','','','','','$10^{-3}$','','',''])

ax1.tick_params(labelsize=9)
ax2.tick_params(labelsize=9)
ax3.tick_params(labelsize=9)
ax4.tick_params(labelsize=9)
ax5.tick_params(labelsize=9)
ax6.tick_params(labelsize=9)
cb1.ax.tick_params(labelsize=9)
cb2.ax.tick_params(labelsize=9)

ax1.set_title('$\\eta=10^{-4}$', fontsize=12, pad=6)
ax2.set_title('$\\eta=10^{-5}$', fontsize=12, pad=6)
ax3.set_title('$\\eta=10^{-6}$', fontsize=12, pad=6)

ax1.set_ylabel('$\\lambda=N^{*}/Q$', fontsize=12, labelpad=23)
ax4.set_ylabel('$\\lambda=N^{*}/Q$', fontsize=12)
cb1.set_label('$w=0$', fontsize=12)
cb2.set_label('$w=1$', fontsize=12)
fig.suptitle('mean immigration rate $\\bar{\\alpha}$', fontsize=12, x=0.53, y=0.06)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fig.savefig('Fig6.svg', format='svg', transparent=True)

plt.show()