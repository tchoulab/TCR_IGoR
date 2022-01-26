#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:23:19 2019

@author: panyunbei
"""

import numpy as np
import matplotlib.pyplot as plt

m0 = open('./Oakes_dataset/fks_CD4_beta_new.txt')
m1 = open('./data_for_modelling/f_k_optimal_deltapi.txt')
m2 = open('./data_for_modelling/f_k_optimal_delta_delta.txt')
s0 = m0.readlines()
s1 = m1.readlines()
s2 = m2.readlines()

data0 = [elem.strip().split('\t') for elem in s0]
data1 = [elem.strip().split('\t') for elem in s1]
data2 = [elem.strip().split('\t') for elem in s2]

k = [int(data1[5][0])]
fks = [float(data0[0][0])]
fksdp4 = [float(data1[5][1])]
fksdp5 = [float(data1[5][2])]
fksdp6 = [float(data1[5][3])]
fksdd4 = [float(data2[5][1])]
fksdd5 = [float(data2[5][2])]
fksdd6 = [float(data2[5][3])]
for i in range(1,len(data0)):
    k.append(int(data1[i+5][0]))
    fks.append(float(data0[i][0]))
    fksdp4.append(float(data1[i+5][1]))
    fksdp5.append(float(data1[i+5][2]))
    fksdp6.append(float(data1[i+5][3]))
    fksdd4.append(float(data2[i+5][1]))
    fksdd5.append(float(data2[i+5][2]))
    fksdd6.append(float(data2[i+5][3]))

n2 = open('./data_for_modelling/fk0001a5.txt') 
p2 = n2.readlines()   
d2 = [elem.strip().split('\t') for elem in p2]
fksw0001 = []
for i in range(len(d2)):
    fksw0001.append(float(d2[i][0]))
    
n0 = open('./data_for_modelling/fk001a5.txt') 
p0 = n0.readlines()   
d0 = [elem.strip().split('\t') for elem in p0]
fksw001 = []
for i in range(len(d0)):
    fksw001.append(float(d0[i][0]))
    
n1 = open('./data_for_modelling/fk005a5.txt') 
p1 = n1.readlines()   
d1 = [elem.strip().split('\t') for elem in p1]
fksw005 = []
for i in range(len(d1)):
    fksw005.append(float(d1[i][0]))

print(fksw001)


fksq2 = [0.878349, 0.000588372, 3.12235*10**(-7), 1.34622*10**(-10), 4.83151*10**(-14), 1.47544*10**(-17), 3.90499*10**(-21), 9.09359*10**(-25), 1.88654*10**(-28), 3.52279*10**(-32), 0,0,0,0,0,0,0,0]
#################################################################
######################### Plot ##################################
    
font1={'weight':'medium', 'size':21}
font2={'weight':'medium', 'size':18}
cmap = plt.get_cmap('jet', 200)

fig,ax0 = plt.subplots(figsize=(6,6))

plt.scatter(k, fks, c='b', label='-CD4-beta')
  
plt.plot(k, fksdp4, c='g', linestyle='--', dashes=((5,5)), label='fksdp, $\eta=10^{-4}$')

plt.plot(k, fksdp5, c='cyan', linestyle='--', dashes=((5,5)), label='fksdp, $\eta=10^{-5}$')

plt.plot(k, fksdp6, c='red', linestyle='--', dashes=((5,5)), label='fksdp, $\eta=10^{-6}$' )

plt.plot(k, fksq2, c='red', label='fksq2, $\eta=10^{-5}$' )
plt.plot(k, fksdd5, c='red', label='fksdd, $\eta=10^{-5}$' )
plt.plot(k, fksdd6, c='firebrick', label='fksdd, $\eta=10^{-6}$' )


#plt.plot(k, fksw0001, c='g', linestyle='--', dashes=((5,5)) ,linewidth=2.5)
#plt.plot(k, fksw001, c='red', linestyle='--', dashes=((5,5)) ,linewidth=2.5)
#plt.plot(k, fksw005, c='cyan', linestyle='--', dashes=((5,5)) ,linewidth=2.5)

#ax0.set_xscale('log')
#ax0.set_yscale('log')

#plt.xlim((1,20))
#plt.ylim((10**(-6), 1.0))
plt.tick_params(labelsize=18)

x = [1,2,3,4,5,6,7,8,9,10,11,12,13]
y = [-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

plt.xticks(x, ('','2','','4','','6','','8','','10','','12',''))
plt.yticks(y, ('','0.0','','0.2','','0.4','','0.6','','0.8','','1.0'))
#plt.yticks(y, ('','$8.6\\time10^{-1}$','','$8.8\\times10^{-1}$','','$9.0\\times10^{-1}$','',
#               '$9.2\\times10^{-1}$','','$9.4\\times10^{-1}$','','$9.6\\times10^{-1}$','',
#               '$9.8\\times10^{-1}$','','$1.0\\times10^{0}$',''))
#plt.legend(fontsize=15, loc=1, bbox_to_anchor=(0.992,0.95))

#plt.grid(which='major')

plt.show()



fig,ax1 = plt.subplots(figsize=(6,6))

plt.scatter(k, fks, c='b', label='-CD4-beta')
  
plt.plot(k, fksdp4, c='g', linestyle='--', dashes=((5,5)), label='fksdp, $\eta=10^{-4}$')

plt.plot(k, fksdp5, c='cyan', linestyle='--', dashes=((5,5)), label='fksdp, $\eta=10^{-5}$')

plt.plot(k, fksdp6, c='red', linestyle='--', dashes=((5,5)), label='fksdp, $\eta=10^{-6}$' )

plt.plot(k, fksdd4, c='red', label='fksdd, $\eta=10^{-4}$' )
plt.plot(k, fksdd5, c='red', label='fksdd, $\eta=10^{-5}$' )
plt.plot(k, fksdd6, c='firebrick', label='fksdd, $\eta=10^{-6}$')

ax1.set_xscale('log')
ax1.set_yscale('log')

plt.tick_params(labelsize=18)

plt.xlim((0.8,20))
plt.ylim((10**(-8), 2.0))
