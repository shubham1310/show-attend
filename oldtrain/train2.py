from core.utils import load_coco_data
import numpy as np
import pickle

data = pickle.load(open('data.txt','r'))
word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
val_data =  pickle.load(open('val_data.txt','r'))

from core.model import CaptionGenerator
from core.solver import CaptioningSolver
model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, prev2out=True, 
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
solver = CaptioningSolver(model, data, val_data, n_epochs=20, batch_size=128, update_rule='adam',
                                          learning_rate=0.001, print_every=1000, save_every=1, image_path='./image/',
                                    pretrained_model=None, model_path='model/lstm/', test_model='model/lstm/model-10',
                                     print_bleu=True, log_path='log/')


solver.train()
