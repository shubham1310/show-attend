import pickle
import numpy as np
path ='/home/msarkar/DynamicAttention/data/'
f= open('/home/code/trainnew_features.pkl','r')
featur = pickle.load(f)
f.close()
caption = np.load(path + 'trainRandomTR_Lbls.npy')


cap=[]
for i in range(len(caption)):
    tmp=[item for sublist in caption[i] for item in sublist]
    while not(len(tmp)==17):
        tmp.append(0)
    cap.append(tmp)

n=len(featur)
feature=[]
for i in range(n):
  feature.append(np.reshape(featur[i], (196,512)))


word_to_idx={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'<START>':10,'<NULL>':11 }

trainn= int(n*0.8)
data= {}
valdata={}
data['captions']= np.array(cap[:trainn])
valdata['captions']= np.array(cap[trainn:])

data['features'] =np.array(feature[:trainn])
valdata['features']=np.array(feature[trainn:])

valdata['image_idxs'] = np.array([i for i in range(trainn)])
data['image_idxs'] = np.array([i for i in range(trainn)])

valdata['file_names'] = [str(i) for i in range(trainn,n)]
data['file_names'] = [str(i) for i in range(trainn)]

data['word_to_idx']=word_to_idx


#cd /home/shubh/code/show-attend/datan/
f = open('data.txt','w')
pickle.dump(data , f)
f.close()
f=  open('valdata.txt','w')
pickle.dump(valdata ,f)
f.close()
