    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import binom

#N_star = 1. * 10**11
#Q = 1. * 10**15
#avg_alpha = 1.6 * 10**(-6) #1.6 * 10**(-8)
#avg_r = 5. * 10**(-4) # 5 * 10**(-4)

#w = np.linspace(6.8 * 10**(-4), 9.8 * 10**(-4), num=4) 

rvals = [5*10**(-4), 10*10**(-4), 25*10**(-4), 50*10**(-4), 75*10**(-4), 90*10**(-4), 100*10**(-4)]

w = [1*10**(-4),2*10**(-4),3*10**(-4),4*10**(-4),5*10**(-4),6*10**(-4),7*10**(-4),8*10**(-4),9*10**(-4),10*10**(-4)]

eta = [2.* 10**(-4), 1.*10**(-3), 2.*10**(-3), 1.*10**(-2), 2.*10**(-2)]

clr0 = ['brown','firebrick','indianred','r','orangered','lightsalmon','navajowhite','yellow','honeydew','aquamarine','darkturquoise','cornflowerblue','darkblue']

clr = ['darkblue','blue','b','dodgerblue','cyan','darkgreen','green','g','limegreen','lime','red','r','darkorange','orange','gold']

#filename = 'plot_KS07_CD8_beta.txt'

###################################################################################################
###################################################################################################

def get_expr_data(gene, chain):
    
    #scriptpath = os.path.dirname(__file__)
    #filename = os.path.join(scriptpath, './naive_memory/dcr_KS07_CD8_naive-2_beta.freq')
    #input_all = open(filename)
    
    input_all_list = []

    if gene == 'CD4':

        if chain == 'alpha':
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD4_naive-1_alpha.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/dcr_KS07_CD4_naive-2_alpha.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/dcr_KS07_CD4_naive-4_alpha.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/vDCRe_alpha_EG10_CD4_naive_alpha.freq', header = None))
        
        if chain == 'beta':
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD4_naive-1_beta.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/dcr_KS07_CD4_naive-2_beta.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/dcr_KS07_CD4_naive-4_beta.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/vDCRe_beta_EG10_CD4_naive_beta.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/vDCRe_beta_SK11_CD4_naive_beta.freq', header = None))

    if gene == 'CD8':

        if chain == 'alpha':
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD8_naive-1_alpha.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/dcr_KS07_CD8_naive-2_alpha.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/dcr_KS07_CD8_naive-3_alpha.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/vDCRe_alpha_EG10_CD8_naive_alpha.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/dcr_alpha_SK11_CD8_naive_alpha.freq', header = None))
        
        if chain == 'beta':
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD8_naive-1_beta.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/dcr_KS07_CD8_naive-2_beta.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/dcr_KS07_CD8_naive-3_beta.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/vDCRe_beta_EG10_CD8_naive_beta.freq', header = None))
            #input_all_list.append(pd.read_csv('./naive_memory/vDCRe_beta_SK11_CD8_naive_beta.freq', header = None))
    
    input_all = pd.concat(input_all_list)
    data_expr = input_all[5].values # numpy array
    
    return input_all[5]

[FONTSIZE_MATH,FONTSIZE_TEXT,FONTSIZE_TICK,FONTSIZE_LEGD] = [30,30,20,24]
[MARGIN_LEFT,MARGIN_RIGHT,MARGIN_BOTTOM,MARGIN_TOP] = [.18,.87,.15,.9]

def get_proliferation_r(i):

    if i==0:
        res = pd.read_csv('./data_for_modelling/ck_r5.txt', sep='\t')
        rowNum = res.shape[0]
    elif i==1:
        res = pd.read_csv('./data_for_modelling/ck_r10.txt', sep='\t')
        rowNum = res.shape[0]
    elif i==2:
        res = pd.read_csv('./data_for_modelling/ck_r25.txt', sep='\t')
        rowNum = res.shape[0]
    elif i==3:
        res = pd.read_csv('./data_for_modelling/ck_r50.txt', sep='\t')
        rowNum = res.shape[0]
    elif i==4:
        res = pd.read_csv('./data_for_modelling/ck_r75.txt', sep='\t')
        rowNum = res.shape[0]
    elif i==5:
        res = pd.read_csv('./data_for_modelling/ck_r90.txt', sep='\t')
        rowNum = res.shape[0]
    else:
        res = pd.read_csv('./data_for_modelling/ck_r100.txt', sep='\t')
        rowNum = res.shape[0]

    ck = [0]
    for j in range(rowNum):
        ck.append(res.loc[j][0])
   
    return ck


def get_proliferation1(i):

    if i==0:
        res = pd.read_csv('./data_for_modelling/ck_w1.txt', sep='\t')
        rowNum = res.shape[0]
    elif i==1:
        res = pd.read_csv('./data_for_modelling/ck_w2.txt', sep='\t')
        rowNum = res.shape[0]
    elif i==2:
        res = pd.read_csv('./data_for_modelling/ck_w3.txt', sep='\t')
        rowNum = res.shape[0]
    elif i==3:
        res = pd.read_csv('./data_for_modelling/ck_w4.txt', sep='\t')
        rowNum = res.shape[0]
    elif i==4:
        res = pd.read_csv('./data_for_modelling/ck_w5.txt', sep='\t')
        rowNum = res.shape[0]
    elif i==5:
        res = pd.read_csv('./data_for_modelling/ck_w6.txt', sep='\t')
        rowNum = res.shape[0]
    elif i==6:
        res = pd.read_csv('./data_for_modelling/ck_w7.txt', sep='\t')
        rowNum = res.shape[0]
    elif i==7:
        res = pd.read_csv('./data_for_modelling/ck_w8.txt', sep='\t')
        rowNum = res.shape[0]
    elif i==8:
        res = pd.read_csv('./data_for_modelling/ck_w9.txt', sep='\t')
        rowNum = res.shape[0]
    else:
        res = pd.read_csv('./data_for_modelling/ck_w10.txt', sep='\t')
        rowNum = res.shape[0]

    ck = [0]
    for j in range(rowNum):
        ck.append(res.loc[j][0])
   
    return ck


def get_proliferation2(i, abun_l):

    if i==0:
        res = pd.read_csv('./data_for_modelling/ck_w1.txt', sep='\t')
    elif i==1:
        res = pd.read_csv('./data_for_modelling/ck_w2.txt', sep='\t')
    elif i==2:
        res = pd.read_csv('./data_for_modelling/ck_w3.txt', sep='\t')
    elif i==3:
        res = pd.read_csv('./data_for_modelling/ck_w4.txt', sep='\t')
    elif i==4:
        res = pd.read_csv('./data_for_modelling/ck_w5.txt', sep='\t')
    elif i==5:
        res = pd.read_csv('./data_for_modelling/ck_w6.txt', sep='\t')
    elif i==6:
        res = pd.read_csv('./data_for_modelling/ck_w7.txt', sep='\t')
    elif i==7:
        res = pd.read_csv('./data_for_modelling/ck_w8.txt', sep='\t')
    elif i==8:
        res = pd.read_csv('./data_for_modelling/ck_w9.txt', sep='\t')
    else:
        res = pd.read_csv('./data_for_modelling/ck_w10.txt', sep='\t')

    ck = [0]
    for j in range(len(abun_l)-1):
        ck.append(res.loc[j][0])
        
    return ck

def get_sampling(eta, cl):

    ck_s = [0.] * len(cl)
    for k in range(1, len(cl)):
        for i in range(1, len(cl)):
            if i >= k:
                ck_s[k] = ck_s[k] + cl[i] * binom.pmf(k,i,eta)
        #print(k,"ck_s",ck_s[k])

    return ck_s

def get_cumulative_0(ck_s):
    
    Cq_si = 0

    for i in range(1, len(ck_s)):
        Cq_si = Cq_si + ck_s[i]
    
    return Cq_si


def get_cumulative_1(ck_s):
    
    Cq_si = 0

    for i in range(1, len(ck_s)):
        Cq_si = Cq_si + ck_s[i] * i
    
    return Cq_si


###################################################################################################
########################################## data ###################################################    
abun_CD4_alpha = get_expr_data(gene='CD4', chain='alpha')
abun_CD8_alpha = get_expr_data(gene='CD8', chain='alpha')
abun_CD4_beta = get_expr_data(gene='CD4', chain='beta')
abun_CD8_beta = get_expr_data(gene='CD8', chain='beta')

print(len(abun_CD4_alpha))
print(len(abun_CD8_alpha))
print(len(abun_CD4_beta))
print(len(abun_CD8_beta))

quit()
abun_list = []
    
abun_list.append(abun_CD4_alpha)
abun_list.append(abun_CD8_alpha)
abun_list.append(abun_CD4_beta)
abun_list.append(abun_CD8_beta)
abun_data = pd.concat(abun_list)
abun = abun_data.values
    
abun_max = np.max(abun)
#print(abun_max)

abun_l = [0] * (abun_max+1)
    
for i in abun:  
    abun_l[i] += 1   

print('finishing abun_l')

#np.savetxt('data.txt',(abun_l_CD4_alpha, abun_l_CD8_alpha, abun_l_CD4_beta, abun_l_CD8_beta), fmt = '%s')
    

################################### function #####################################################
ck = []
for i in range(len(w)):
    ck.append(get_proliferation1(i))

print('finishing ck')
    
cl = []
for i in range(len(w)):
    cl.append(get_proliferation2(i, abun_l))

print('finishing cl')

cl_r = []
for i in range(len(rvals)):
    cl_r.append(get_proliferation_r(i))

print('finishing cl_r')
 
#################################### normalization ##################################################

l = []
for i in range(len(abun_l)):
    l.append(i)
    
l = l[1:]

abun_l = abun_l[1:]
    
print('normalizing 1')

ck_new = []
for x in ck:
    x = x[1:]
    ck_new.append(x)

cl_new = []
for x in cl:
    x = x[1:]
    cl_new.append(x)

cl_r_new = []
for x in cl_r:
    x = x[1:]
    cl_r_new.append(x)
    


###################################################################################################
############################################## plot ###############################################

cmap = plt.get_cmap('jet')

wvals = [3*10**(-4), 6*10**(-4), 8*10**(-4), 10**(-3), 2*10**(-3), 3*10**(-3), 4*10**(-3), 5*10**(-3), 6*10**(-3), 7*10**(-3), 8*10**(-3),9*10**(-3), 10*10**(-3)]

w1 = [0.35*10**(-4),1*10**(-4),2*10**(-4),3*10**(-4),4*10**(-4),5*10**(-4),6*10**(-4),7*10**(-4),8*10**(-4),9*10**(-4),9.8*10**(-4)]

#fig = plt.figure(figsize=(10.5,7.5))

font1={'weight':'medium',
        'size':11}
########################################################################
fig, ax0 = plt.subplots()

r_bar = 4.9*10**(-4)
pi_r0 = []
for n in w1:
    pi_i_r0 = []
    x = np.linspace(0, 1*10**(-3),500000)
    
    for i in x:
        if abs(i-r_bar) < n/2:
            pi_i_r0.append(1/n)
        else:
            pi_i_r0.append(0)
    
    pi_r0.append(pi_i_r0)

N0 = len(w1)
cmap0 = plt.get_cmap('jet', N0)

for i in range(len(pi_r0)):
    plt.plot(x, pi_r0[i], c=cmap0(i), linewidth=1)

plt.xlim((-0.5*10**(-4), 1.1*10**(-3)))  # TODO
plt.ylim((-2000, 3.1*10**4)) 
plt.tick_params(labelsize=10)

plt.xlabel('Proliferation rate $r$ [$day^{-1}$]', font1)
plt.ylabel('distributions $\pi_r(r)$', font1) 

#ax0.xaxis.get_major_formatter().set_powerlimits((0,1))
#ax0.yaxis.get_major_formatter().set_powerlimits((0,1))

x=[0.0, 1*10**(-4),2*10**(-4),3*10**(-4),4*10**(-4),5*10**(-4),6*10**(-4),7*10**(-4),8*10**(-4),9*10**(-4),10*10**(-4)]

plt.xticks(x, ('0.0','','','','','$0.5\\times10^{-3}$','','','','','$1.0\\times 10^{-3}$'))

#plt.gcf().set_facecolor(np.ones(3)* 240 / 255)   
plt.grid(which='major')

###############################################
#axins = inset_axes(ax1, width= 2, height=2, loc=1)

axins = fig.add_axes([0.65, 0.6, 0.22, 0.22])
#cm = plt.cm.get_cmap('RdYlBu')
pi_mean = [r_bar for i in w1] 

pi_std = []
for i in w1:
    pi_std.append((1/N0 * (i)**2)**(1/2))
#
plt.scatter(w1, pi_mean, c=w1, s=12, marker ='+', cmap=cmap)
plt.scatter(w1, pi_std, c=w1, s=6, cmap=cmap)

axins.annotate("mean[$\pi_r$]", (0.315, 0.65),
                xycoords="axes fraction",fontsize = 10) 
axins.annotate("std[$\pi_r$]", (0.55, 0.15),
                xycoords="axes fraction",fontsize = 10) 

plt.xlim((0, 11*10**(-4)))  # TODO
plt.ylim((0, 0.6*10**(-3))) 
plt.tick_params(labelsize=7)

plt.xlabel('width $w$', fontsize=10)

x1=[0.0, 2*10**(-4), 4*10**(-4), 6*10**(-4), 8*10**(-4), 10*10**(-4)]
y1=[0.0,1*10**(-4),2*10**(-4),3*10**(-4),4*10**(-4),5*10**(-4),6*10**(-4)]
plt.xticks(x1, ('0.0','','','','','$1.0\\times 10^{-3}$'))
plt.yticks(y1, ('0.0','','$0.2\\times10^{-3}$','','$0.4\\times 10^{-3}$','','$0.6\\times10^{-3}$'))

#plt.show()

##########################################################################

fig, ax1 =plt.subplots()

N1 = len(ck_new)

cmap1 = plt.get_cmap('jet', N1)

for i in range(len(ck_new)):
    row = range(1,len(ck_new[i])+1)
    ax1.plot(row, ck_new[i], c=cmap1(i), linewidth=1) 

ax1.set_xscale('log')
ax1.set_yscale('log')

plt.xlim((0.5, 20000))  # TODO
plt.ylim((10**0, 10**11)) 
plt.tick_params(labelsize=10)

plt.xlabel('clone size $k$', font1)
plt.ylabel('clone count $c_k$', font1) 


#plt.gcf().set_facecolor(np.ones(3)* 240 / 255)  
plt.grid(which='major')

norm = mpl.colors.Normalize(vmin=1*10**(-4),vmax=10*10**(-4))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar1 = plt.colorbar(sm)
cbar1.set_label('width $w$', rotation=90, fontsize=10)

#plt.show()

############################################################################
fig, ax2 = plt.subplots()

pi_r = []
for m in wvals:
    pi_i_r = []
    x = np.linspace(0, 10*10**(-3),100000)
    
    for i in x:
        if i < m:
            pi_i_r.append(1/m)
        else:
            pi_i_r.append(0)
    
    pi_r.append(pi_i_r)

N2 = len(wvals)
cmap2 = plt.get_cmap('jet', N2)

for i in range(len(pi_r)):
    plt.plot(x, pi_r[i], c=cmap2(i), linewidth=1)


plt.xlim((0, 1.2*10**(-2)))  # TODO
plt.ylim((-200, 4*10**3)) 
plt.tick_params(labelsize=10)

plt.xlabel('Proliferation rate $r$ [$day^{-1}$]', font1)
plt.ylabel('distributions $\pi_r(r)$', font1)  

plt.gcf().set_facecolor(np.ones(3)* 240 / 255)   
plt.grid()

###############################################
#axins = inset_axes(ax1, width= 2, height=2, loc=1)

axins1 = fig.add_axes([0.65, 0.6, 0.22, 0.22])

pi_mean = [x/2 for x in wvals]
plt.scatter(wvals, pi_mean, c=pi_mean, s=6, cmap=cmap)

plt.xlim((0.0, 1.1*10**(-2)))  # TODO
plt.ylim((1*10**(-4), 6*10**(-3))) 
plt.tick_params(labelsize=7)

plt.xlabel('width $w$', fontsize=10)

x2=[0.0, 25*10**(-4), 50*10**(-4), 75*10**(-4), 100*10**(-4)]
y2=[0.0, 2*10**(-3), 4*10**(-3), 6*10**(-3)]

plt.xticks(x2, ('0.0','','','','$1.0\\times10^{-2}$'))
plt.yticks(y2, ('0.0','$0.2\\times10^{-2}$','$0.4\\times10^{-2}$','$0.6\\times10^{-2}$'))

axins1.annotate("mean[$\pi_r$]", (0.5, 0.3),
                xycoords="axes fraction",fontsize = 8) 
axins1.annotate("std[$\pi_r$]", (0.5, 0.15),
                xycoords="axes fraction",fontsize = 10) 

############################################################################

fig, ax3 =plt.subplots()
 #(figsize=(8,6))

N3 = len(cl_r_new)
cmap3 = plt.get_cmap('jet', N3)

for i in range(len(cl_r_new)):
    row = range(1,len(cl_r_new[i])+1)
    ax3.plot(row, cl_r_new[i], c=cmap3(i), linewidth=1) 

ax3.set_xscale('log')
ax3.set_yscale('log')

plt.xlim((0.5, 20000))  
plt.ylim((10**0, 10**11)) 
plt.tick_params(labelsize=10)

plt.xlabel('clone size $k$', font1)
plt.ylabel('clone count $c_k$', font1) 

sm = plt.cm.ScalarMappable(cmap=cmap)
cbar2 = plt.colorbar(sm)
cbar2.set_label('width $w$', rotation=90, fontsize=10)

plt.gcf().set_facecolor(np.ones(3)* 240 / 255)  
plt.grid()

#plt.subplots_adjust(wspace =0.45, hspace =0.45)

plt.show()

