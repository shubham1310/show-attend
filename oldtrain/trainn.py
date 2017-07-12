from core.solver2 import CaptioningSolver
from core.model1 import CaptionGenerator

import pickle
import argparse
import numpy as np

def main():
    
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('-bs', '--batch_size', default= 64, type=int, help="Batch size",)
    parser.add_argument('-pt','--path', default='./datan/', help="Path to a data file")
    parser.add_argument('-dim1','--dim_feature1', required = True, help="Dimension", type=int)
    parser.add_argument('-ep','--epoch', default=1000, help="Number of epochs", type=int)
    parser.add_argument('-pe', '--print_every', default=100, type=int)
    parser.add_argument('-se','--save_every', default=10, type=int)
    args = parser.parse_args()

    batch_size =  args.batch_size
    dim_feature1=args.dim_feature1
    path = args.path
    epoch=args.epoch
    print_every=args.print_every
    save_every=args.save_every

    
    # load train dataset
    path ='/home/msarkar/DynamicAttention/data/'
    f= open('/home/shubh/code/trainnew_features.pkl','r')
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
      feature.append(np.reshape(featur[i], (dim_feature1,512)))
    del featur


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


    # f=open(path+'data.txt', 'r')
    # data = pickle.load(f)
    # f.close()
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    # val_data = pickle.load(open('./datan/valdata.txt', 'r'))

    model = CaptionGenerator(word_to_idx, dim_feature=[dim_feature1, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, prev2out=True, 
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, valdata, n_epochs=epoch, batch_size=batch_size, update_rule='adam',
                                          learning_rate=0.001, print_every=print_every, save_every=save_every,
                                    pretrained_model=None, model_path='model/lstm' +str(dim_feature1) +'/', test_model='model/lstm' +str(dim_feature1) +'/model-10',
                                     print_bleu=True, log_path='log' +str(dim_feature1) +'/')

    solver.train()

if __name__ == "__main__":
    main()