#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:13:44 2019

@author: panyunbei
"""

    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from openpyxl import Workbook


###################################################################################################
###################################################################################################

def get_expr_data(gene, chain):
    
    input_all_list = []
    
    if gene == 'CD4':
            
        if chain == 'alpha':
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD4_naive-1_alpha.freq', header = None)[5])
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD4_naive-2_alpha.freq', header = None)[5])
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD4_naive-4_alpha.freq', header = None)[5])
            input_all = pd.concat(input_all_list)
            input_all_list = []
            input_all_list.append(input_all)
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/vDCRe_alpha_SK11_CD4_naive_alpha.freq', header = None)[5])
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/vDCRe_alpha_EG10_CD4_naive_alpha.freq', header = None)[5])
            
        if chain == 'beta':
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD4_naive-1_beta.freq', header = None)[5])
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD4_naive-2_beta.freq', header = None)[5])
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD4_naive-4_beta.freq', header = None)[5])
            input_all = pd.concat(input_all_list)
            input_all_list = []
            input_all_list.append(input_all)
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/vDCRe_beta_EG10_CD4_naive_beta.freq', header = None)[5])
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/vDCRe_beta_SK11_CD4_naive_beta.freq', header = None)[5])

            
    if gene == 'CD8':
            
        if chain == 'alpha':
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD8_naive-1_alpha.freq', header = None)[5])
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD8_naive-2_alpha.freq', header = None)[5])
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD8_naive-3_alpha.freq', header = None)[5])
            input_all = pd.concat(input_all_list)
            input_all_list = []
            input_all_list.append(input_all)
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/vDCRe_alpha_EG10_CD8_naive_alpha.freq', header = None)[5])
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_alpha_SK11_CD8_naive_alpha.freq', header = None)[5])
            
            
        if chain == 'beta':
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD8_naive-1_beta.freq', header = None)[5])
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD8_naive-2_beta.freq', header = None)[5])
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/dcr_KS07_CD8_naive-3_beta.freq', header = None)[5])
            input_all = pd.concat(input_all_list)
            input_all_list = []
            input_all_list.append(input_all)
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/vDCRe_beta_EG10_CD8_naive_beta.freq', header = None)[5])
            input_all_list.append(pd.read_csv('./Oakes_dataset/naive_memory/vDCRe_beta_SK11_CD8_naive_beta.freq', header = None)[5])

                 
    return input_all_list


###################################################################################################
###################################################################################################

def mean_value(ck):
    
    mean_val = [0] * np.max([np.max(len(ck[0])),np.max(len(ck[1])),
               np.max(len(ck[2]))])
    for i in ck:
         for j in range(len(i)):
             mean_val[j] += i[j]/len(ck)

    return mean_val

def mean_clone_counts(ck):
    
    total_cells = [] * len(ck)
    num_clones = [np.sum(ck[0]),np.sum(ck[1]),np.sum(ck[2])]
    
    for j in range(len(ck)):
        for i in range(len(j)):
            total_cells = total_cells + i * j[i]

    return total_cells/num_clones
        
def fk(ck):
    
    ck1 = []
    for i in range(len(ck)):
        ck1.append(ck[i]*i)
    
    ck2 = []
    for i in ck1:
        ck2.append(i/np.sum(ck1))

    return ck2

def Fq(ck):

    F = [0]
    for i in range(1,len(ck)):
        F.append(F[i-1]+ck[i])

    return F

def statistics(ck):
    
    mean_val = mean_value(ck)
    
    sem = [0] * len(mean_val)
    cv = [0] * len(mean_val)
    for j in ck:
        for i in range(len(sem)):
            if i > len(j)-1:
                sem[i] += (mean_val[i]-0)**2 / 5
            else:
                sem[i] += (mean_val[i]-j[i])**2 / 5
    
    for i in range(len(sem)):
        sem[i] = (sem[i]/4)**(1/2)
    
    for i in range(len(cv)):
        if mean_val[i] == 0:
            cv[i] = 0
        else:
            cv[i] = (sem[i])**(1/2) / mean_val[i]   
        
    return sem, cv
    
###################################################################################################
###################################################################################################
abun_CD4_alpha = get_expr_data(gene='CD4',chain='alpha')
abun_CD4_beta = get_expr_data(gene='CD4',chain='beta')
abun_CD8_alpha = get_expr_data(gene='CD8',chain='alpha')
abun_CD8_beta = get_expr_data(gene='CD8',chain='beta')

#print(abun_CD4_alpha)
#print(abun_CD8_alpha)

abun_l_CD4_alpha = []
for i in abun_CD4_alpha:
    abun_l_CD4_alpha.append([0] * (np.max(i)+1))

abun_l_CD4_beta = []
for i in abun_CD4_beta:
    abun_l_CD4_beta.append([0] * (np.max(i)+1))

abun_l_CD8_alpha = []
for i in abun_CD8_alpha:
    abun_l_CD8_alpha.append([0] * (np.max(i)+1))

abun_l_CD8_beta = []
for i in abun_CD8_beta:
    abun_l_CD8_beta.append([0] * (np.max(i)+1))
    
#print(abun_l_CD4_alpha)
   
for i in range(len(abun_CD4_alpha)):  
    for j in abun_CD4_alpha[i]:
        abun_l_CD4_alpha[i][j] += 1

for i in range(len(abun_CD4_beta)):  
    for j in abun_CD4_beta[i]:
        abun_l_CD4_beta[i][j] += 1
        
for i in range(len(abun_CD8_alpha)):  
    for j in abun_CD8_alpha[i]:
        abun_l_CD8_alpha[i][j] += 1
        
for i in range(len(abun_CD8_beta)):
    for j in abun_CD8_beta[i]:
        abun_l_CD8_beta[i][j] += 1  
  
#print(abun_l_CD4_alpha)

fk_CD4_alpha = []
for i in abun_l_CD4_alpha:
    fk_CD4_alpha.append(fk(i))

fk_CD4_beta = []
for i in abun_l_CD4_beta:
    fk_CD4_beta.append(fk(i))
    
fk_CD8_alpha = []
for i in abun_l_CD8_alpha:
    fk_CD8_alpha.append(fk(i))
    
fk_CD8_beta = []
for i in abun_l_CD8_beta:
    fk_CD8_beta.append(fk(i))

print(fk_CD4_alpha)   

for i in range(len(abun_l_CD4_alpha)):
    abun_l_CD4_alpha[i] = abun_l_CD4_alpha[i][1:]

for i in range(len(abun_l_CD4_beta)):
    abun_l_CD4_beta[i] = abun_l_CD4_beta[i][1:]
    
for i in range(len(abun_l_CD8_alpha)):
    abun_l_CD8_alpha[i] = abun_l_CD8_alpha[i][1:]
    
for i in range(len(abun_l_CD8_beta)):
    abun_l_CD8_beta[i] = abun_l_CD8_beta[i][1:]

#print(abun_l_CD4_alpha)

mean_clone_counts_CD4_alpha = mean_value(abun_l_CD4_alpha)
mean_clone_counts_CD4_beta = mean_value(abun_l_CD4_beta)
mean_clone_counts_CD8_alpha = mean_value(abun_l_CD8_alpha)
mean_clone_counts_CD8_beta = mean_value(abun_l_CD8_beta)
print(mean_clone_counts_CD4_alpha)
print(mean_clone_counts_CD4_beta)
print(mean_clone_counts_CD8_alpha)
print(mean_clone_counts_CD8_beta)

for i in range(len(fk_CD4_alpha)):
    fk_CD4_alpha[i] = fk_CD4_alpha[i][1:]

for i in range(len(fk_CD4_beta)):
    fk_CD4_beta[i] = fk_CD4_beta[i][1:]
    
for i in range(len(fk_CD8_alpha)):
    fk_CD8_alpha[i] = fk_CD8_alpha[i][1:]
    
for i in range(len(fk_CD8_beta)):
    fk_CD8_beta[i] = fk_CD8_beta[i][1:]

#np.savetxt('KS07_fks_CD4_alpha.txt',(fk_CD4_alpha),fmt='%s')
#np.savetxt('KS07_fks_CD4_beta.txt',(fk_CD4_beta),fmt='%s')
#np.savetxt('KS07_fks_CD8_alpha.txt',(fk_CD8_alpha),fmt='%s')
#np.savetxt('KS07_fks_CD8_beta.txt', (fk_CD8_beta), fmt='%s')
#print(fk_CD4_alpha)

SEM_CD4_alpha, CV_CD4_alpha= statistics(fk_CD4_alpha)
SEM_CD4_beta, CV_CD4_beta = statistics(fk_CD4_beta)
SEM_CD8_alpha, CV_CD8_alpha = statistics(fk_CD8_alpha)
SEM_CD8_beta, CV_CD8_beta = statistics(fk_CD8_beta)   

print(SEM_CD4_alpha)

np.savetxt('CD4_alpha.txt',(SEM_CD4_alpha, CV_CD4_alpha),fmt='%s')
np.savetxt('CD4_beta.txt',(SEM_CD4_beta, CV_CD4_beta),fmt='%s')
np.savetxt('CD8_alpha.txt',(SEM_CD8_alpha, CV_CD8_alpha),fmt='%s')
np.savetxt('CD8_beta.txt', (SEM_CD8_beta, CV_CD8_beta), fmt='%s')