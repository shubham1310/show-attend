from core.solvers.solvernpy import CaptioningSolver
from core.models.modelslim import CaptionGenerator

import pickle
import argparse
import numpy as np,cv2,os

def main():
    # load train dataset
    path ='/home/msarkar/DynamicAttention/data/'
    caption = np.load(path + 'trainRandomTR_Lbls.npy')
    npath = path +'images/'

    cap=[]
    n = len(caption)
    # for i in range(len(caption)):
    for i in range(n):
        tmp=[item for sublist in caption[i] for item in sublist]
        tmp =  [10] + tmp
        while not(len(tmp)==18):
            tmp.append(11)
        cap.append(tmp)


    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('-bs', '--batch_size', default= 64, type=int, help="Batch size",)
    parser.add_argument('-dim1','--dim_feature1', required = True, help="Dimension", type=int)
    parser.add_argument('-ep','--epoch', default=1000, help="Number of epochs", type=int)
    parser.add_argument('-pe', '--print_every', default=200, type=int)
    parser.add_argument('-se','--save_every', default=10, type=int)
    parser.add_argument('-pt','--pretrain', default=None)
    parser.add_argument('-fl','--flag', default=1, type=int)
    parser.add_argument('-nm','--name', default='')
    args = parser.parse_args()

    batch_size =  args.batch_size
    dim_feature1=args.dim_feature1
    # path = args.path
    epoch=args.epoch
    print_every=args.print_every
    save_every=args.save_every
    pretrain=args.pretrain
    flag = args.flag
    name = args.name

    # parser.add_argument('-pt','--pretrain', default=None)

    word_to_idx = word_to_idx={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,
                                '7':7,'8':8,'9':9,'<START>':10,'<NULL>':11 }

    model = CaptionGenerator(word_to_idx, dim_feature=[dim_feature1, 512], dim_embed=512,
                                    dim_hidden=1024, n_time_step=17, prev2out=False, imshape=128,
                                    ctx2out=True, alpha_c=1.0, selector=True, dropout=True,channels=1)

    solver = CaptioningSolver(model, caption=np.array(cap), path=path, n_epochs=epoch, batch_size=batch_size, update_rule='adam',
                        learning_rate=0.001, print_every=print_every, save_every=save_every, pretrained_model=pretrain,
                        model_path='model/'+ name+'dec' +str(dim_feature1)+'_' + str(batch_size) +'/',
                        test_model='model/'+ name+'dec' +str(dim_feature1)+'_'+ str(batch_size)  +'/model-10',
                        print_bleu=True, log_path='log/'+ 'dec' +str(dim_feature1)+'_' + str(batch_size)  +'/')
    
    solver.train(tp='slim')

if __name__ == "__main__":
    main()