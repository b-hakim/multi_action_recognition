from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch
from utils import *
from dataloader import spatial_dataloader
from constants import EXPERIMENTS


def fusion_accuracry(rgb_preds, opf_preds, path, train_path, test_path):

    with open(rgb_preds,'rb') as f:
        rgb =pickle.load(f)
    f.close()
    with open(opf_preds,'rb') as f:
        opf =pickle.load(f)
    f.close()

    method = EXPERIMENTS.MAIN_AUTHOR

    dataloader = spatial_dataloader(BATCH_SIZE=1,
                                    num_workers=1,
                                    path=path,
                                    trainfile =train_path,
                                    testfile=test_path,
                                    experiment=method,
                                    step_size=1)

    train_loader,val_loader,test_video = dataloader.run()

    video_level_preds = np.zeros((len(rgb.keys()),dataloader.max_cls_ind))
    video_level_labels = np.zeros(len(rgb.keys()))
    correct=0
    ii=0
    for name in sorted(rgb.keys()):   
        r = rgb[name]
        o = opf[name]

        label = int(test_video[name])-1
                    
        video_level_preds[ii,:] = (r+o)
        
        video_level_labels[ii] = label
        ii+=1         
        if np.argmax(r+o) == (label):
            correct+=1

    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_preds = torch.from_numpy(video_level_preds).float()
        
    top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))     
                                
    print top1,top5

if __name__ == '__main__':
    rgb_preds = 'record/spatial/stabilized_side_view_spatial_video_preds.pickle'
    opf_preds = 'record/motion/stabilized_side_view_motion_video_preds.pickle'
    ds_path = '/home/bassel/data/office-actions/office_actions_19/short_clips/stabilized_resized_frms_224/'
    trainfile = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_trainlist.txt'
    testfile = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_testlist.txt'

    fusion_accuracry(rgb_preds, opf_preds, ds_path, trainfile, testfile)