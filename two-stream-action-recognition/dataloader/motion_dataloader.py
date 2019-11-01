import numpy as np
import pickle
from PIL import Image
import time
import shutil
import random
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from split_train_test_video import *
import constants
import glob


class motion_dataset(Dataset):  
    def __init__(self, dic, in_channel, root_dir, mode, transform=None, step_size=1, experiment=constants.EXPERIMENTS.MAIN_AUTHOR):
        #Generate a 16 Frame clip
        self.keys=dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.transform = transform
        self.mode=mode
        self.in_channel = in_channel
        self.img_rows=224
        self.img_cols=224
        self.step_size=step_size
        self.experiment=experiment

    def stackopf(self):
        u = self.root_dir + "step_" + str(self.step_size) + "/" + self.video
        v = self.root_dir + "step_" + str(self.step_size) + "/" + self.video
        
        flow = torch.FloatTensor(2*self.in_channel,self.img_rows,self.img_cols)
        if self.experiment == constants.EXPERIMENTS.MAIN_AUTHOR:
            # Author method
            i = int(self.clips_idx)
            start = 0
            end = self.in_channel-1
            step = 1
        elif self.experiment == constants.EXPERIMENTS.SUMMARIZE_VIDEO_10X_10Y_CHANNELS:
            # Our method
            size = self.in_channel
            start = 1
            step = (self.clips_idx-start)/(size-1)
            # assert step == 1
            end = 1 + (size - 1) * step
        elif self.experiment == constants.EXPERIMENTS.MULTIPLE_STEPS__CLIPS_START_STEP_END:
            prefix, start, step, end = self.clips_idx.split()
            prefix, start, step, end = int(prefix), int(start), int(step), int(end)

        for j, idx in enumerate(range(start, end+1, step)):

            if self.experiment == constants.EXPERIMENTS.MAIN_AUTHOR:
                idx = i + j

            h_image = u + '/{}_flow_x_{:05}.jpg'.format(prefix, idx)
            v_image = v + '/{}_flow_y_{:05}.jpg'.format(prefix, idx)

            imgH=(Image.open(h_image))
            imgV=(Image.open(v_image))

            H = self.transform(imgH)
            V = self.transform(imgV)
            
            flow[2*(j),:,:] = H
            flow[2*(j)+1,:,:] = V

            imgH.close()
            imgV.close()

        return flow

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)

        if self.experiment == constants.EXPERIMENTS.MAIN_AUTHOR:
        #Author Method
            if self.mode == 'train':
                self.video, nb_clips = self.keys[idx].split('##')

                # Author method
                self.clips_idx = random.randint(1,int(nb_clips))
            elif self.mode == 'val':
                self.video,self.clips_idx = self.keys[idx].split('##')
            else:
                raise ValueError('There are only train and val mode')
        elif self.experiment == constants.EXPERIMENTS.SUMMARIZE_VIDEO_10X_10Y_CHANNELS:
            # Our Method
            self.video, nb_clips = self.keys[idx].split('##')
            self.clips_idx = int(nb_clips.split(";")[0])
        elif self.experiment == constants.EXPERIMENTS.MULTIPLE_STEPS__CLIPS_START_STEP_END:
            self.video, nb_clips = self.keys[idx].split('##')
            self.clips_idx = nb_clips

        label = self.values[idx]
        label = int(label)-1 # as label is guarantteed to be 1 based
        data = self.stackopf()

        if self.mode == 'train':
            sample = (data,label)
        elif self.mode == 'val':
            sample = (self.video,data,label)
        else:
            raise ValueError('There are only train and val mode')
        return sample


class Motion_DataLoader():
    def __init__(self, BATCH_SIZE, num_workers, in_channel, path, trainfile, testfile,
                 step_size, experiment, ucf_format=False, is_label_zero_based=False):
        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.frame_count={}
        self.in_channel = in_channel
        self.data_path=path
        # split the training and testing videos
        splitter = Office_Actions_splitter(trainfile, testfile, is_label_zero_based)
        self.train_video, self.test_video, self.max_cls_ind = splitter.split_video()
        self.step_size = step_size
        self.experiment = experiment
        self.ucf_format = ucf_format

    def generate_frame_count(self, dump_pckle_path):
        all_videos_dir = self.data_path+"/step_"+str(self.step_size)
        dic = {};

        for video in os.listdir(all_videos_dir):
            num_frames = ""

            for step_i in range(self.step_size):
                step_frames = glob.glob(os.path.join(all_videos_dir, video, "{}_*".format(step_i)))
                num_frames += str(len(step_frames)/2)+";"
                # print (os.path.join(all_videos_dir, video), str(num_frames))
                # assert num_frames == 10

            dic[video.replace(".mp4","")] = num_frames

        pickle.dump(dic, open(dump_pckle_path, 'wb'))

    def generate_frame_count_ucf_format(self, dump_pckle_path):
        all_categories_dir = self.data_path+"/step_"+str(self.step_size)
        dic = {};

        for category in os.listdir(all_categories_dir):
            all_videos_for_current_cat = os.listdir(os.path.join(all_categories_dir, category))

            for video in all_videos_for_current_cat:
                num_frames = ""

                for step_i in range(self.step_size):
                    step_frames = glob.glob(os.path.join(all_categories_dir,
                                                         category, video,
                                                         "{}_*".format(step_i)))

                    num_frames += str(len(step_frames)/2)+";"
                    # print (os.path.join(all_videos_dir, video), str(num_frames))
                    # assert num_frames == 10

                dic[category+"/"+video.replace(".mp4","")] = num_frames

        pickle.dump(dic, open(dump_pckle_path, 'wb'))

    def load_frame_count(self):
        #print '==> Loading frame number of each video'
        pickle_path = 'dataloader/dic/motion_frame_count.pickle'

        if not os.path.exists(pickle_path):
            if self.ucf_format:
                self.generate_frame_count_ucf_format(pickle_path)
            else:
                self.generate_frame_count(pickle_path)

        with open(pickle_path,'rb') as file:
            self.frame_count = pickle.load(file)

        # for line in dic_frame :
        #     line2 = line.replace('HandstandPushups', 'HandStandPushups')
            # videoname = line.split('_',1)[1].split('.',1)[0]
            # n,g = videoname.split('_',1)
            # if n == 'HandStandPushups':
            #     videoname = 'HandstandPushups_'+ g
            # self.frame_count[videoname]=dic_frame[line]
            # self.frame_count[line2] = dic_frame[line]

    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample19()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader, self.test_video
            
    def val_sample19(self):
        self.dic_test_idx = {}
        #print len(self.test_video)
        for video in self.test_video:
            n,g = video.split('_',1)
            if self.experiment == constants.EXPERIMENTS.MAIN_AUTHOR:
                # Author method
                sampling_interval = int((int(self.frame_count[video].split(";")[0])-10+1)/19)
                for index in range(19):
                    nb_clips = index*sampling_interval
                    nb_clips = nb_clips + 1
                # nb_clips = random.randint(1, self.frame_count[video]-10+1)
                    key = video + '##' + str(nb_clips)
                    self.dic_test_idx[key] = self.test_video[video]
            elif self.experiment == constants.EXPERIMENTS.SUMMARIZE_VIDEO_10X_10Y_CHANNELS:
                #Our method
                nb_clips = self.frame_count[video]
                key = video + '##' + str(nb_clips)
                self.dic_test_idx[key] = self.test_video[video]
            elif self.experiment == constants.EXPERIMENTS.MULTIPLE_STEPS__CLIPS_START_STEP_END:
                nb_clips_per_start = self.frame_count[video].split(';')

                # assert  nb_clips == 120 # for this kind of experiments
                for i in range(self.step_size):
                    nb_clips = int(nb_clips_per_start[i])
                    nb_blocks = nb_clips / (self.in_channel*self.step_size)

                    for b in range(nb_blocks):
                        # for start in range(b*self.in_channel+1, (b+1)*self.in_channel+1):
                            # actual_start = b*self.step_size*self.in_channel +start # for #blocks = 4 for step 3, actual start for block 1(2nd) is (1*3*10+1) for 4th >> 3*3*10+1 = 91
                            # end = actual_start + self.step_size*(self.in_channel-1)
                            # key = video + '##' + str(start-1) + " " + str(actual_start)+" "+ str(self.step_size) + " " + str(end)
                            # self.dic_test_idx[key] = self.test_video[video]
                        start = b*self.in_channel+1
                        end = (b+1)*self.in_channel
                        key = video + '##' + str(i) + " " + str(start)+" "+ str(self.step_size) + " " + str(end)
                        self.dic_test_idx[key] = self.test_video[video]

            # elif self.experiment == constants.EXPERIMENTS.MULTIPLE_STEPS_ALL_STACKS:


    def get_training_dic(self):
        self.dic_video_train={}

        for video in self.train_video:
            video = video.replace(".mp4", "")
            if self.experiment == constants.EXPERIMENTS.MAIN_AUTHOR:
                nb_frames = int(self.frame_count[video].split(";")[0])-10+1
                key = video +'##' + str(nb_frames)
                self.dic_video_train[key] = self.train_video[video]
            elif self.experiment == constants.EXPERIMENTS.SUMMARIZE_VIDEO_10X_10Y_CHANNELS:
                nb_frames = self.frame_count[video]
                key = video + '##' + str(nb_frames)
                self.dic_video_train[key] = self.train_video[video]
            elif self.experiment == constants.EXPERIMENTS.MULTIPLE_STEPS__CLIPS_START_STEP_END:
                nb_clips_per_start = self.frame_count[video].split(';')

                # assert  nb_clips == 120 # for this kind of experiments
                for i in range(self.step_size):
                    nb_clips = int(nb_clips_per_start[i])
                    nb_blocks = nb_clips / (self.in_channel * self.step_size)

                    for b in range(nb_blocks):
                        start = b * self.in_channel + 1
                        end = (b + 1) * self.in_channel
                        key = video + '##' + str(i) + " " + str(start) + " " + str(self.step_size) + " " + str(end)
                        self.dic_video_train[key] = self.train_video[video]

                # nb_frames = self.frame_count[video]
                # # assert nb_frames == 120  # for this kind of experiments
                # nb_blocks = nb_frames / (self.in_channel * self.step_size)
                #
                # for b in range(nb_blocks):
                #     for start in range(1, self.step_size + 1):
                #         actual_start = b * self.step_size * self.in_channel + start  # for #blocks = 4 for step 3, actual start for block 1(2nd) is (1*3*10+1) for 4th >> 3*3*10+1 = 91
                #         end = actual_start + self.step_size * (self.in_channel - 1)
                #         key = video + '##' + str(start-1) + " " + str(actual_start) + " " + str(self.step_size) + " " + str(end)
                #         self.dic_video_train[key] = self.train_video[video]

    def train(self):
        training_set = motion_dataset(dic=self.dic_video_train, in_channel=self.in_channel, root_dir=self.data_path,
                                    mode='train', step_size=self.step_size,
                                    transform = transforms.Compose([
                                            transforms.Scale([224,224]),
                                            transforms.ToTensor()]),
                                    experiment=self.experiment
                                    )
        # print '==> Training data :',len(training_set),' videos',training_set[1][0].size()

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
            )

        return train_loader

    def val(self):
        validation_set = motion_dataset(dic= self.dic_test_idx, in_channel=self.in_channel, root_dir=self.data_path ,
                                        mode ='val', step_size=self.step_size,
                                        transform = transforms.Compose([
                                            transforms.Scale([224,224]),
                                            transforms.ToTensor(),
                                        ]),
                                        experiment=self.experiment
                                        )
        # print '==> Validation data :',len(validation_set),' frames',validation_set[1][1].size()
        #print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader

if __name__ == '__main__':
    data_loader =Motion_DataLoader(BATCH_SIZE=1,num_workers=1,in_channel=10,
                                   path='/home/bassel/data/UCF101_optical_flow/step1/',
                                   ucf_list='/home/bassel/data/ucfTrainTestlist/',
                                        ucf_split='01'
                                        )
    train_loader,val_loader,test_video = data_loader.run()
    #print train_loader,val_loader