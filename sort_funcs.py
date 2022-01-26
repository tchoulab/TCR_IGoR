
import heapq
import os, time, path
import pandas as pd
import csv
import numpy as np

############################################# sorting algorithms ##################################################
res = pd.read_csv('example_pgens_1e5_nt.tsv', sep='\t',header = None)
data = res.values
print(data[:,1])
print(np.max(data[:,1]))

#for i in range(len(re)):
#    j = i
#    a = re[i]
#    print(a)
#    while (j < len(re)-2) and (re[j,1] < re[j+1,1]):
#        re[j] = re[j+1]
#        re[j+1] = a
#        print('re',re[j+1])
#        j += 1
    #print(re)


data = data[np.argsort(-data[:, 1])]
print(data)

csvFile = open('example_pgens_1e5_nt.tsv', "w")           
writer = csv.writer(csvFile)                                             
writer.writerow(["nt_sequences","Pgens_nt", "aa_sequences", "Pgens_aa"])
writer.writerows(data)
csvFile.close()

