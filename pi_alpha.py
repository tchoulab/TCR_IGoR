
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from scipy.optimize import curve_fit  
from scipy import asarray as ar, exp
from scipy import log 
from sympy import *

alpha_bar = 1.6 * 10**(-8)
the = [0.5]

clr = ['darkblue', 'blue', 'b', 'dodgerblue', 'cyan', 'darkgreen', 'g', 'green', 'limegreen', 'lime', 'red', 'r', 'darkorange', 'orange', 'gold']


[FONTSIZE_MATH, FONTSIZE_TEXT, FONTSIZE_TICK, FONTSIZE_LEGD] = [20, 20, 16, 16]
[MARGIN_LEFT, MARGIN_RIGHT, MARGIN_BOTTOM, MARGIN_TOP] = [.18, .87, .15, .9]


############################################### function ###############################################3####
def get_immigration(i):

    res = pd.read_csv('pi_theta45.txt', sep='\t')
    rowNum = res.shape[0]

    ck = [0]
    for j in range(rowNum):
        ck.append(res.loc[j][0])
   
    return ck


def sum_pi(alpha_i):
    
    pi = [0]*len(alpha_i)
    
    pi[0] = alpha_i[0]
    for i in range(1, len(alpha_i)):
        pi[i] = pi[i-1] + alpha_i[i]

    #print(pi)
    for i in range(1, len(alpha_i)):
        pi[i] = pi[i]/pi[-1]
        
    return pi

def get_PT():
    
    #filename = 'example_pgens_'+str(i)+'.tsv'
        
    df =  pd.read_csv('example_pgens_1e5_nt.tsv', sep = ',')

    Q = [50, 75, 100, 125, 150, 175, 200, 225, 250, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
    
    P_T = []

    for i in Q:
        
        P_gen = [df.values[:, 1][x] for x in range(i)]   
        P_T.append(np.sum(P_gen))
        
    #print(Q)
    #print(P_T)

    return Q, P_T    

##############################################################################################################

df =  pd.read_csv('example_pgens_1e5_nt.tsv', sep = ',')
#print(df)
#df.columns=["aa_sequnce","Pgen_estimate"]
#df.to_csv('example_pgens_10e5.tsv', index=False)
#data = df.sort_values('Pgen_estimate', ascending = False, inplace = False)

P_gen = df.values[:, 1]
P_total = np.sum(P_gen)

P = [0.] * len(P_gen)
for i in range(len(P_gen)):
    P[i] = P_gen[i]/P_total

alpha_i = [0.] * len(P_gen)
for i in range(len(P_gen)):
    alpha_i[i] = P_gen[i]/P_total * 1.6 * 10**7

alpha_i_inv = alpha_i[::-1]

k = list(range(1, len(alpha_i)+1, 1))

ck = []
for i in range(len(the)):
    ck.append(get_immigration(i))
     
    
ck_new = []
for x in ck:
    x = x[1:]

    ck_new.append(x)

#print(len(ck_new[0]))

sum_al = sum_pi(alpha_i_inv)
#print(Pi_al)

Q, P_T = get_PT()

##########################################################################################################
############################################ plot ########################################################

def func(x, a, b, c, d):
    #return (-a) * np.exp(-b * x) + c
    #return (-a) * np.log(b * x) + c
    #return (-a) * pow(2, -c * x) + d
    return a + b * x**(-c)

fig, ax6 = plt.subplots(figsize=(8, 6))   
x = np.array(alpha_i_inv)
y = np.array(sum_al)

popt, pcov = curve_fit(func, x, y)
print(popt)

plt.plot(x, y, 's', label='original values')                     # culmulation of \alpha_j versus \alpha_j
plt.plot(x, func(x, *popt), 'r', label='polyfit values')         # data fitting (use power law)
plt.xlabel('$\\alpha_j$', fontsize=18)
plt.ylabel('$\sum_{i>j} \\alpha_i$', fontsize=18)


###############################################

fig, ax5 = plt.subplots(figsize=(8, 6))

r = Symbol("r")
dify = diff(func(r, *popt), r, 1)
print(dify)

yvals = []
for i in x:
    yvals.append(dify.subs('r', i))

plt.plot(x, yvals)

plt.xlabel('$\\alpha$', fontsize=18)
plt.ylabel('$\\pi_\\alpha(\\alpha)$', fontsize=18)               # \pi_alpha versus \alpha_j


plt.gcf().set_facecolor(np.ones(3)* 240 / 255)  
plt.grid()

#plt.legend()
#plt. show()

#quit()

################################################

fig, ax2 = plt.subplots(figsize=(8, 6))

ax2.scatter(alpha_i_inv, sum_al, marker='v', cmap='red', alpha=0.3) 

x1 = alpha_i_inv[9954]
y1 = sum_al[9954]
print((x1, y1))

plt.plot([x1, x1,], [0, y1], 'k--', linewidth=1.5)
plt.plot([0, x1,], [y1, y1], 'k--', linewidth=1.5)

plt.xlim(0, 3*10**5)  # TODO
plt.ylim(0, 2*10**7) 

plt.scatter([x1,], [y1, ], s=50, color='r')

x1 = round(x1, 1)
y1 = round(y1, 1)
plt.annotate(r'For immigration rate $\alpha_j=%s1$, $\sum_{i>j}\alpha_i=%s2$'%(x1, y1), xy=(x1, y1), xycoords='data', xytext=(+5, -45),
             textcoords='offset points', fontsize=12,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.3"))

plt.xlabel('$\\alpha_j$', fontsize=18)
plt.ylabel('$\sum_{i>j} \\alpha_i$', fontsize=18)

ax2.yaxis.label.set_size(16)
ax2.xaxis.label.set_size(16)

plt.gcf().set_facecolor(np.ones(3)* 240 / 255)  
plt.grid()

#plt.legend()
#plt.show()

#################################################

fig, ax1 = plt.subplots(figsize=(8, 6))

for i in range(len(ck_new)):
    row = list(range(1, len(ck_new[i])+1))
    ax1.plot(row, ck_new[i], label='$\\theta=$'+str(the[i]), color=clr[i]) 

ax1.set_xscale('log')
ax1.set_yscale('log')

plt.xlim((0.05, 4000))  # TODO
plt.ylim((10**0, 10**11)) 

plt.xlabel('$k$', fontsize=18)
plt.ylabel('$c_k$', fontsize=18)

ax1.yaxis.label.set_size(16)
ax1.xaxis.label.set_size(16)

plt.gcf().set_facecolor(np.ones(3)* 240 / 255)   
plt.grid()

#plt.legend()
#plt.show()

####################################################

fig, ax0 = plt.subplots(figsize=(8, 6))

alpha = np.arange(0, 3*10**(-8), 10**(-10))
pi_alpha = [0.] * (len(alpha))
for i in range(len(theta)):
    for j in range(len(alpha)):
        if alpha[j] >= alpha_bar*(1-theta[i]):
            pi_alpha[j] =((alpha_bar * (1-theta[i]))**(1/theta[i]))/(theta[i]*alpha[j]**(1+1/theta[i]))
        else:
            pi_alpha[j] = 0
    
    ax0.plot(alpha, pi_alpha, color=clr[i])

plt.xlim((0, 3.5*10**(-8)))  # TODO
plt.ylim((0.0, 1.35*10**9)) 

plt.xlabel('immigration rate $\\alpha_j$', fontsize=18)
plt.ylabel('distribution $\pi_{\\alpha}$', fontsize=18)

ax0.yaxis.label.set_size(16)
ax0.xaxis.label.set_size(16)

#plt.show()

quit()
################################################

fig, ax4 = plt.subplots(figsize=(8, 6))

ax4.scatter(Q, P_T, marker='v', cmap='red', alpha=0.75)

plt.xlim((25, 3275))  # TODO
plt.ylim(0, 5*10**(-3)) 

ax4.yaxis.label.set_size(16)
ax4.xaxis.label.set_size(16)

plt.xlabel('Q', fontsize=18)
plt.ylabel('Total Probability $P_T$', fontsize=18)               # Total Probability P_T versus clone rank i
plt.gcf().set_facecolor(np.ones(3)* 240 / 255)   
plt.grid()

#plt.legend()
#plt.show()

#quit()

################################################

fig, ax3 = plt.subplots(figsize=(8, 6))

ax3.scatter(k, P, alpha=0.85, s=15) 

x0 = k[59]
y0 = P[59]

plt.plot([x0, x0,], [0, y0], 'k--', linewidth=1.5)
plt.plot([0, x0,], [y0, y0], 'k--', linewidth=1.5)

plt.scatter([x0,], [y0, ], s=50, color='r')

y0 = round(y0, 5)
plt.annotate(r'for clone rank j=60, the corresponding probability $P_j=%s$'% y0, xy=(x0, y0), xycoords='data', xytext=(-75, +135),
             textcoords='offset points', fontsize=12,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.3"))

plt.xlim(0, 200)  # TODO
plt.ylim(0, 2*10**(-2)) 

ax3.yaxis.label.set_size(16)
ax3.xaxis.label.set_size(16)

plt.xlabel('clone rank i', fontsize=18)
plt.ylabel('probability P', fontsize=18)                         # probability P versus clone rank i
plt.gcf().set_facecolor(np.ones(3)* 240 / 255)  
plt.grid()

#plt.legend()
#plt.show()

