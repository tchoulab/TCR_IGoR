import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

if os.path.isfile('freqa_count_fixed.bin'):
	with open('freqa_count_fixed.bin','rb') as f1:
		freqs_count = pickle.load(f1)

print(freqs_count)









