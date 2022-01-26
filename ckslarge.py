#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:13:42 2019

@author: panyunbei
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

m = []
m.append(open('./data_for_modelling/cklarge_unnormalized_a2.txt'))
m.append(open('./data_for_modelling/cklarge_unnormalized_b2.txt'))
m.append(open('./data_for_modelling/cklarge_unnormalized_c2.txt'))
m.append(open('./data_for_modelling/cklarge_unnormalized_d2.txt'))


s = []
for i in m:
    s.append(i.readlines())

data = []
for i in s:
    data.append([elem.strip().split('\t') for elem in i])

x = []
x1 = np.arange(1,301,1)
x2 = np.arange(310,1010,10)
x3 = np.arange(1200,10200,200)

x.extend(x1)
x.extend(x2)
x.extend(x3)


col1 = []
col2 = []
col3 = []
col4 = []
col5 = []
col6 = []
for i in range(3):
    for j in range(1,len(data[i])):
        col1.append(float(data[i][j][0]))
        col2.append(float(data[i][j][1]))
        col3.append(float(data[i][j][2]))
        col4.append(float(data[i][j][3]))
        col5.append(float(data[i][j][4]))
        col6.append(float(data[i][j][5]))
        
for i in range(3,len(data)):
    for j in range(1,len(data[i])):
        x.append(float(data[i][j][0]))
        col1.append(float(data[i][j][1]))
        col2.append(float(data[i][j][2]))
        col3.append(float(data[i][j][3]))
        col4.append(float(data[i][j][4]))
        col5.append(float(data[i][j][5]))
        col6.append(float(data[i][j][6]))

#################################################################
######################### Plot ##################################
    
font1={'weight':'medium', 'size':21}
font2={'weight':'medium', 'size':18}
cmap = plt.get_cmap('jet', 1000)

fig,ax0 = plt.subplots(figsize=(6,6))

plt.plot(x, col1, c=cmap(0), label='$w=0.0$')
plt.plot(x, col2, c=cmap(25), label='$w=0.025$')
plt.plot(x, col3, c=cmap(50), label='$w=0.05$' )
plt.plot(x, col4, c=cmap(75), label='$w=0.075$' )
plt.plot(x, col5, c=cmap(100), label='$w=0.1$' )
plt.plot(x, col6, c=cmap(999), label='$w=1$' )

plt.xlim((0.8,10**5))
plt.ylim((10**(0), 10**10))
plt.tick_params(labelsize=18)

ax0.set_xscale('log')
ax0.set_yscale('log')

plt.legend(fontsize=17, loc=1, bbox_to_anchor=(0.995,0.995))
#sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#cbar1 = plt.colorbar(sm)
#cbar1.set_label('width $w$', rotation=90, fontsize=18)
#cbar1.ax.tick_params(labelsize=18)
#plt.grid(which='major')
#plt.grid(True)

x = [0.8,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,
     200,300,400,500,600,700,800,900,10**3,2*10**3,3*10**3,4*10**3,5*10**3,
     6*10**3,7*10**3,8*10**3,9*10**3,10**4,2*10**4,3*10**4,4*10**4,5*10**4,
     6*10**4,7*10**4,8*10**4,9*10**4,10**5]
y = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,
     200,300,400,500,600,700,800,900,10**3,2*10**3,3*10**3,4*10**3,5*10**3,
     6*10**3,7*10**3,8*10**3,9*10**3,10**4,2*10**4,3*10**4,4*10**4,5*10**4,
     6*10**4,7*10**4,8*10**4,9*10**4,10**5,2*10**5,3*10**5,4*10**5,5*10**5,
     6*10**5,7*10**5,8*10**5,9*10**5,10**6,2*10**6,3*10**6,4*10**6,5*10**6,
     6*10**6,7*10**6,8*10**6,9*10**6,10**7,2*10**7,3*10**7,4*10**7,5*10**7,
     6*10**7,7*10**7,8*10**7,9*10**7,10**8,2*10**8,3*10**8,4*10**8,5*10**8,
     6*10**8,7*10**8,8*10**8,9*10**8,10**9,2*10**9,3*10**9,4*10**9,5*10**9,
     6*10**9,7*10**9,8*10**9,9*10**9,10**10]
#     6*10**10,7*10**10,8*10**10,9*10**10,10**11]
plt.xticks(x, ('','$10^0$','','','','','','','','','$10^1$','','','','','','','','','$10^2$'
               ,'','','','','','','','','$10^3$','','','','','','','','','$10^4$','','','','',
               '','','','','$10^5$'))
plt.yticks(y, ('$10^0$','','','','','','','','','','','','','','','','','','$10^2$'
               ,'','','','','','','','','','','','','','','','','','$10^4$',
               '','','','','','','','','','','','','','','','','','$10^6$',
               '','','','','','','','','','','','','','','','','','$10^8$',
               '','','','','','','','','','','','','','','','','','$10^{10}$'))


plt.show()