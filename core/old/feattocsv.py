import csv, numpy as np
import pickle
f= open('/home/shubh/code/trainnew_features.pkl','r')
featur = pickle.load(f)
f.close()

wr=csv.writer(open('features.csv','w'))
for i in range(len(featur)):
    tmp = featur[i].flatten()
    wr.writerow(tmp)
