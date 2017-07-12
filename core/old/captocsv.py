import csv, numpy as np
path ='/home/msarkar/DynamicAttention/data/'
caption = np.load(path + 'trainRandomTR_Lbls.npy')

wr=csv.writer(open('captions.csv','w'))
cap=[]
for i in range(len(caption)):
    tmp=[item for sublist in caption[i] for item in sublist]
    while not(len(tmp)==17):
        tmp.append(0)
    wr.writerow(tmp)
