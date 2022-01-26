
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit  
from scipy import log,exp

#with open('example_seqs_5e7_1.tsv') as csvfile:

    #df = csv.reader(csvfile, delimiter='\t')

    #data = np.array(list(df))
    #nt_seqs = list(data[:,0])

#seqs_num = len(nt_seqs)

#print('finishing reading')

#freq = []
#for i in nt_seqs:
#    freq.append(nt_seqs.count(i)/seqs_num)

#print('finishing adding')

#freq = sorted(freq ,reverse = True)

#print('finishing sorting')

#print(freq)










#with open("alpha_seqs_1e8.tsv") as csvfile1:

#    reader1 = csv.reader(csvfile1, delimiter = '\t')
#    dic = {}

#    for line in reader1:
#        seq = line[0]
#        if seq in dic.keys():
#            dic[seq] += 1
#        else:
#            dic[seq] = 1

#print('finish reading')

#freq = dic.values()
#freq = list(freq)



m = open('alpha.txt')
s = m.readlines()

data = [elem.strip().split(' ') for elem in s]

freq1 = data[1]
C = len(freq1)
print(C)

freq1 = [int(i) for i in freq1]

freq = []
for i in range(10000):
    freq.append(int(freq1[i]))

freq = np.array(freq)

sum1 = np.sum(freq)
print(sum1)

for i in freq1:
    i = i/(10**8)

for i in freq:
    i = i/sum1

print('finish reading')

freq1 = sorted(freq1, reverse = True)
freq = sorted(freq, reverse = True)
print('finish sorting')

###################################################################
###################################################################
font1={'weight':'medium', 'size':21}
font2={'weight':'medium', 'size':12}


def f(x,a,b,c):
    return a*exp(-b*x)/(x**c)

fig,ax0 = plt.subplots(figsize=(6,6))

plt.plot(range(1,len(freq1)+1),freq1,'k')

#xdata = np.array(range(1,len(freq)+1))
#ydata = np.array(freq)

#xdata1 = np.array(range(1,len(freq1)+1))

#popt, pcov = curve_fit(f, xdata, ydata)
#print(popt)

#plt.plot(xdata1, f(xdata1, *popt), 'r', '-', label='fitting values')

ax0.set_xscale('log')
ax0.set_yscale('log')

plt.xlim((0.5, 1.5*10**8))
plt.ylim((0.5, 10000))# TODO
plt.tick_params(labelsize=18)

x=[0.5,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10**4,2*10**4,3*10**4,4*10**4,5*10**4,6*10**4,7*10**4,8*10**4,9*10**4,10**5,2*10**5,3*10**5,4*10**5,5*10**5,6*10**5,7*10**5,8*10**5,9*10**5,10**6,2*10**6,3*10**6,4*10**6,5*10**6,6*10**6,7*10**6,8*10**6,9*10**6,10**7,2*10**7,3*10**7,4*10**7,5*10**7,6*10**7,7*10**7,8*10**7,9*10**7,10**8,1.5*10**8]

plt.xticks(x, ('','$10^0$','','','','','','','','','','','','','','','','','','$10^2$','','','','','','','','','','','','','','','','','','$10^4$','','','','','','','','','','','','','','','','','','$10^6$','','','','','','','','','','','','','','','','','','$10^8$',''))

plt.xlabel('clone rank', font1)
plt.ylabel('frequency', font1)

##########################

axins1 = fig.add_axes([0.24, 0.2, 0.268, 0.268])

plt.plot(range(1,len(freq)+1),freq, 'k')

#xdata1 = np.array(range(1,len(freq)+1))
#ydata = np.array(freq)

#popt, pcov = curve_fit(f, xdata, ydata)
#print(popt)

#plt.plot(xdata, f(xdata, *popt), 'r-', label='fitting values')

#plt.xlabel('clone rank $i$', font2)
#plt.ylabel('frequency $f_i$', font2)

plt.xlim((-200, 3000))
plt.ylim((0, 6000))# TODO
plt.tick_params(labelsize=13)

x1 = [0,1000,2000,3000]
y1=[0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]

plt.xticks(x1, ('0', '', '', '3000'))
plt.yticks(y1,('0','','','','2000','','','','4000','','','','6000'))

plt.show()
