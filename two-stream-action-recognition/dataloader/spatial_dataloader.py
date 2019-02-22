import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from split_train_test_video import *
from skimage import io, color, exposure

class spatial_dataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform=None):
 
        self.keys = dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.mode =mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def load_ucf_image(self,video_name, index):
        # if video_name.split('_')[0] == 'HandstandPushups':
        #     n,g = video_name.split('_',1)
        #     name = 'HandstandPushups_'+g
        #     path = self.root_dir + 'HandstandPushups'+'/v_'+name+'/{:05}.jpg'.format(index)
        # else:
        path = self.root_dir + video_name+'/{:04}.jpg'.format(index)
         
        img = Image.open(path)
        transformed_img = self.transform(img)
        img.close()

        return transformed_img

    def __getitem__(self, idx):

        if self.mode == 'train':
            video_name, nb_clips = self.keys[idx].split(' ')
            nb_clips = int(nb_clips)
            clips = []
            clips.append(random.randint(1, nb_clips/3))
            clips.append(random.randint(nb_clips/3, nb_clips*2/3))
            clips.append(random.randint(nb_clips*2/3, nb_clips+1))
            
        elif self.mode == 'val':
            video_name, index = self.keys[idx].split(' ')
            index =abs(int(index))
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1
        
        if self.mode=='train':
            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                data[key] = self.load_ucf_image(video_name, index)
                    
            sample = (data, label)
        elif self.mode=='val':
            data = self.load_ucf_image(video_name,index)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path, trainfile, testfile, step_size, experiment):
        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.frame_count ={}
        # split the training and testing videos
        splitter = Office_Actions_splitter(trainfile, testfile)
        self.train_video, self.test_video, self.max_cls_ind = splitter.split_video()
        self.step_size = step_size
        self.experiment = experiment


    def generate_frame_count(self, dump_pckle_path):
        all_videos_dir = self.data_path
        dic = {};

        for video in  os.listdir(all_videos_dir):
            num_frames = len(os.listdir(os.path.join(all_videos_dir, video)))/2
            print (os.path.join(all_videos_dir, video), str(num_frames))
            # assert num_frames == 10

            dic[video.replace(".mp4","")] = num_frames;

        pickle.dump(dic, open(dump_pckle_path, 'wb'))

    def load_frame_count(self):
        #print '==> Loading frame number of each video'
        pickle_path = 'dataloader/dic/spatial_frame_count.pickle'

        if not os.path.exists(pickle_path):
            self.generate_frame_count(pickle_path)

        with open(pickle_path,'rb') as file:
            dic_frame = pickle.load(file)

        self.frame_count = dic_frame

    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample20()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    def get_training_dic(self):
        #print '==> Generate frame numbers of each training video'
        self.dic_training={}
        for video in self.train_video:
            #print videoname
            nb_frame = self.frame_count[video]-10+1
            key = video+' '+ str(nb_frame)
            self.dic_training[key] = self.train_video[video]
                    
    def val_sample20(self):
        print '==> sampling testing frames'
        self.dic_testing={}

        for video in self.test_video:
            nb_frame = self.frame_count[video]-10+1
            interval = int(nb_frame/19)

            for i in range(19):
                frame = i*interval
                key = video+ ' '+str(frame+1)
                self.dic_testing[key] = self.test_video[video]

    def train(self):
        training_set = spatial_dataset(dic=self.dic_training, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                # transforms.RandomCrop(224),
                transforms.Scale([224, 224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print '==> Training data :',len(training_set),'frames'
        print training_set[1][0]['img1'].size()

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        validation_set = spatial_dataset(dic=self.dic_testing, root_dir=self.data_path, mode='val',
                                         transform = transforms.Compose([
                            transforms.Scale([224,224]),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                         ]))
        
        print '==> Validation data :',len(validation_set),'frames'
        print validation_set[1][1].size()

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader


if __name__ == '__main__':
    
    dataloader = spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
                                path='/home/bassel/data/UCF101/',
                                ucf_list='/home/bassel/data/ucfTrainTestlist/',
                                ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()