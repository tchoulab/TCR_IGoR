
import numpy as np
import matplotlib.pyplot as plt
from math import log

m = []
#m.append(open('./data_for_modelling/cksmall_unnormalized_a.txt'))
#m.append(open('./data_for_modelling/cksmall_unnormalized_b.txt'))
#m.append(open('./data_for_modelling/cksmall_unnormalized_c.txt'))
#m.append(open('./data_for_modelling/cksmall_unnormalized_d.txt'))
#m.append(open('./data_for_modelling/cksmall_unnormalized_e.txt'))
#m.append(open('./data_for_modelling/ckq2unsam0.txt'))
#m.append(open('./data_for_modelling/ckq2unsam1.txt'))
#m.append(open('./data_for_modelling/ckq2unsam2.txt'))
#m.append(open('./data_for_modelling/ckq2unsam3.txt'))
m.append(open('./data_for_modelling/ckq11.txt'))
m.append(open('./data_for_modelling/ckq20.txt'))

s = []
for i in m:
    s.append(i.readlines())

data = []
for i in s:
    data.append([elem.strip().split('\t') for elem in i])

x = np.arange(1,301,1)
#x2 = np.arange(310,2010,10)
#x3 = np.arange(2100,6100,100)
#x4 = np.arange(6000,31500,500)
#x5 = np.arange(31000,103000.0,2000)

#x = []
#x.extend(x1)
#x.extend(x2)
#x.extend(x3)
#x.extend(x4)
#x.extend(x5)

col = []
for i in data:
    coli = []
    for j in range(1,len(i)):
        coli.append(float(i[j][0]))
    col.append(coli)

#for i in range(2):
#    for j in range(1,len(data[i])):
#        col1.append(float(data[i][j][0]))
#        col2.append(float(data[i][j][1]))
#        col3.append(float(data[i][j][2]))
#        col4.append(float(data[i][j][3]))
#        col5.append(float(data[i][j][4]))
#        col6.append(float(data[i][j][5]))

#for i in range(2,len(data)):
#    for j in range(1,len(data[i])):
#        x.append(float(data[i][j][0]))
#        col1.append(float(data[i][j][1]))
#        col2.append(float(data[i][j][2]))
#        col3.append(float(data[i][j][3]))
#        col4.append(float(data[i][j][4]))
#        col5.append(float(data[i][j][5]))
#        col6.append(float(data[i][j][6]))



#################################################################
######################### Plot ##################################
    
font1={'weight':'medium', 'size':21}
font2={'weight':'medium', 'size':18}
cmap = plt.get_cmap('jet', 1000)

fig,ax0 = plt.subplots(figsize=(6,6))

plt.plot(x, col[0], c=cmap(99), label='$q=1$')
plt.plot(x, col[1], c=cmap(899), label='$q=2$')
#plt.plot(x, col[2], c=cmap(399), label='$w=0.4$' )
#plt.plot(x, col[3], c=cmap(599), label='$w=0.6$' )
#plt.plot(x, col[4], c=cmap(799), label='$w=0.8$' )
#plt.plot(x, col[5], c=cmap(999), label='$w=1$' )

plt.xlim((0.8,200))
plt.ylim((10**(-20), 10**(-3)))
plt.tick_params(labelsize=18)

ax0.set_xscale('log')
ax0.set_yscale('log')

#plt.grid(which='major')

#y = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,
#     200,300,400,500,600,700,800,900,10**3,2*10**3,3*10**3,4*10**3,5*10**3,
#     6*10**3,7*10**3,8*10**3,9*10**3,10**4,2*10**4,3*10**4,4*10**4,5*10**4,
#     6*10**4,7*10**4,8*10**4,9*10**4,10**5,2*10**5,3*10**5,4*10**5,5*10**5,
#     6*10**5,7*10**5,8*10**5,9*10**5,10**6,2*10**6,3*10**6,4*10**6,5*10**6,
#     6*10**6,7*10**6,8*10**6,9*10**6,10**7,2*10**7,3*10**7,4*10**7,5*10**7,
#     6*10**7,7*10**7,8*10**7,9*10**7,10**8,2*10**8,3*10**8,4*10**8,5*10**8,
#     6*10**8,7*10**8,8*10**8,9*10**8,10**9,2*10**9,3*10**9,4*10**9,5*10**9,
#     6*10**9,7*10**9,8*10**9,9*10**9,10**10]
#plt.yticks(y, ('$10^0$','','','','','','','','','','','','','','','','','','$10^2$'
#               ,'','','','','','','','','','','','','','','','','','$10^4$',
#               '','','','','','','','','','','','','','','','','','$10^6$',
#               '','','','','','','','','','','','','','','','','','$10^8$',
#               '','','','','','','','','','','','','','','','','','$10^{10}$'))

plt.legend(fontsize=18, loc=1, bbox_to_anchor=(0.99,0.99))

plt.show()