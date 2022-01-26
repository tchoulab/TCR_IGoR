import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

if os.path.isfile('freqa_count_fixed.bin'):
	with open('freqa_count_fixed.bin','rb') as f1:
		freqs_count = pickle.load(f1)

#while freqs_count[-1] == 0:
#	freqs_count.pop(-1)

#with open('freqb_count_fixed.bin','wb') as f2:
#	pickle.dump(freqs_count, f2, pickle.HIGHEST_PROTOCOL)

y = []
length = []
for i in range(len(freqs_count)):
	if freqs_count[i] != 0:
		y.append(i)
		length.append(freqs_count[i])
y = y[::-1]
y.append(1)
length = length[::-1]

x = [length[0]]
for i in range(len(length)):
	x.append(x[i-1]+length[i])


fig, ax0 = plt.subplots(figsize=(6,6))
ax0.plot(x,y,'k')
ax0.set_xscale('log')
ax0.set_yscale('log')

plt.xlim((0.5, 1.5*10**9))
plt.ylim((0.5, 100000))
plt.tick_params(labelsize=18)

x1 = [0.5,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10**4,2*10**4,3*10**4,4*10**4,5*10**4,6*10**4,7*10**4,8*10**4,9*10**4,10**5,2*10**5,3*10**5,4*10**5,5*10**5,6*10**5,7*10**5,8*10**5,9*10**5,10**6,2*10**6,3*10**6,4*10**6,5*10**6,6*10**6,7*10**6,8*10**6,9*10**6,10**7,2*10**7,3*10**7,4*10**7,5*10**7,6*10**7,7*10**7,8*10**7,9*10**7,10**8,2*10**8,3*10**8,4*10**8,5*10**8,6*10**8,7*10**8,8*10**8,9*10**8,10**9,1.5*10**9]

plt.xticks(x1, ('','$10^0$','','','','','','','','','','','','','','','','','','$10^2$','','','','','','','','','','','','','','','','','','$10^4$','','','','','','','','','','','','','','','','','','$10^6$','','','','','','','','','','','','','','','','','','$10^8$','','','','','','','','','',''))


axins1 = fig.add_axes([0.24, 0.2, 0.268, 0.268])

plt.plot(x,y, 'k')

#xdata1 = np.array(range(1,len(freq)+1))
#ydata = np.array(freq)

#popt, pcov = curve_fit(f, xdata, ydata)
#print(popt)

#plt.plot(xdata, f(xdata, *popt), 'r-', label='fitting values')

#plt.xlabel('clone rank $i$', font2)
#plt.ylabel('frequency $f_i$', font2)

plt.xlim((-200, 3000))
plt.ylim((0, 60000))# TODO
plt.tick_params(labelsize=13)

x2 = [0,1000,2000,3000]
y2=[0,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000]

plt.xticks(x2, ('0', '', '', '3000'))
plt.yticks(y2,('0','','','','20000','','','','40000','','','','60000'))


plt.show()









