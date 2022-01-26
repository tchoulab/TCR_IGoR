import numpy as np


f = open('outa09.txt','r')
m = f.readlines()

outfile = open('output09.txt','w')

freq = 1
i = 0
while i < len(m):
	if i == len(m)-1:
		string = m[i].strip() +'\t'+str(freq)+'\n'
		outfile.write(string)
		i += 1
	elif m[i] == m[i+1]:
		freq += 1
		i += 1
	else:
		string = m[i].strip() +'\t'+str(freq)+'\n'
		outfile.write(string)/Users/yunbei/Dropbox/My Mac (ip-192-168-43-22.ap-northeast-1.compute.internal)/Desktop/figures
		freq = 1
		i += 1

outfile.close()

