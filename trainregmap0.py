from core.solverregmap import CaptioningSolver
from core.modelregmap import CaptionGenerator

import pickle
import argparse
import numpy as np,cv2,os

def main():
    # load train dataset
    path ='/home/msarkar/DynamicAttention/data/'
    bbox = np.load(path + 'trainRandomTR_BBxs1.npy')

    
    n = len(bbox)
    # for i in range(len(caption)):
    bboxes = []
    for i in range(n):
        tmp=bbox[i]
        # if not(len(tmp)==1):
        #     continue
        # print tmp
        tmp =[np.array([0.0,0.0,0.0,0.0]), tmp[0]]
        # print tmp

        # while not(len(tmp)==2):
        #     tmp.append(np.array([-1.0,-1.0,-1.0,-1.0]))
        bboxes.append(tmp)
    # print bboxes[:10]
    # for i in range(n):
    #     for j in bboxes[i]:
    #         if not(len(j)==4):
    #             print bboxes[i]

    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('-bs', '--batch_size', default= 64, type=int, help="Batch size",)
    parser.add_argument('-dim1','--dim_feature1', required = True, help="Dimension", type=int)
    parser.add_argument('-ep','--epoch', default=1000, help="Number of epochs", type=int)
    parser.add_argument('-pe', '--print_every', default=200, type=int)
    parser.add_argument('-se','--save_every', default=10, type=int)
    parser.add_argument('-op','--optimizer', default='adam')
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
    update_rule=args.optimizer
    pretrain=args.pretrain
    flag = args.flag
    name = args.name
    print update_rule

    model = CaptionGenerator( dim_feature=[dim_feature1, 512], batch_size=batch_size, dim_embed=512,
                                    dim_hidden=1024, n_time_step=1, prev2out=True, imshape=128,
                                    ctx2out=True, alpha_c=1.0, selector=True, flag=flag , dropout=True,channels=1)

    solver = CaptioningSolver(model, bbox=np.array(bboxes), path=path, n_time_step=1,  n_epochs=epoch, batch_size=batch_size, update_rule=str(update_rule),
                        learning_rate=0.001, print_every=print_every, save_every=save_every, pretrained_model=pretrain,
                        model_path='model/lstm'+ 'regmap' + name + update_rule +str(dim_feature1)+'_' + str(batch_size) +'/',
                        test_model='model/lstm'+ 'regmap' + name + update_rule +str(dim_feature1)+'_'+ str(batch_size)  +'/model-10',
                        print_bleu=True, log_path='log/'+ 'regmap'+ name + update_rule  +str(dim_feature1)+'_' + str(batch_size)  +'/')
    
    solver.train()

if __name__ == "__main__":
    main()