import threading

import numpy as np
import cv2
import glob
import os
import pdb


class ThreaddatasetMeanCalculator (threading.Thread):
    def __init__(self, threadID, ds_dir, ds_train_lst):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.ds_dir = ds_dir
        self.ds_train_lst = ds_train_lst
        self.sum=0
        self.count=0
        self.std_sum=0

    def run(self):
        self.sum, self.std_sum, self.count = calculate_thread_dataset_mean(self.ds_dir, self.ds_train_lst)


def calculate_thread_dataset_mean(ds_dir, ds_train_lst, num_frames=16, new_w_h_size=112):
    sum = np.zeros((new_w_h_size, new_w_h_size, 3))
    std_sum = np.zeros((3))
    count = 0
    mean = [ 91.86219342,  98.20372432, 103.10237516]

    for line in ds_train_lst:
        vid_path = line.strip().split()[0].replace(".mp4", "").replace(".avi", "").split("/")[-1].replace("HandStandPushups", "HandstandPushups")
        # start_pos = int(line.split()[1])
        # lbl = int(line.split()[2])
        start_pos = 1
        num_frames = len(os.listdir(os.path.join(ds_dir, vid_path)))
        stack_frames = []

        for i in range(start_pos, start_pos+num_frames):
            img = cv2.imread(os.path.join(ds_dir, vid_path, "{:04}.jpg".format(i)))
            # img = cv2.resize(img, (new_w_h_size, new_w_h_size))
            if img is None:
                print os.path.join(ds_dir, vid_path, "{:04}.jpg".format(i))
            height, width, _ = img.shape

            if height != new_w_h_size or width != new_w_h_size:
                if (width > height):
                    scale = float(new_w_h_size) / float(height)
                else:
                    scale = float(new_w_h_size) / float(width)

                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

                height, width, _ = img.shape

                crop_y = int((height - new_w_h_size) / 2)
                crop_x = int((width - new_w_h_size) / 2)

                # stack_frames.append(img[crop_y:crop_y + new_w_h_size, crop_x:crop_x + new_w_h_size, :])
                sum += img[crop_y:crop_y + new_w_h_size, crop_x:crop_x + new_w_h_size, :]
                diff = img[crop_y:crop_y + new_w_h_size, crop_x:crop_x + new_w_h_size, :]-mean
                std_sum += (diff * diff).sum(axis=0).sum(axis=0)
            else:
                # stack_frames.append(img)
                sum+=img
                std_sum += ((img-mean)*(img-mean)).sum(axis=0).sum(axis=0)


            count += 1

            # stack_frames.append(img)

        # stack_frames = np.array(stack_frames)
        # sum += stack_frames
        # count += 1
    # mean/=float(count)
    # print mean
    return sum, std_sum, count


def calculate_dataset_mean(ds_dir, ds_list, num_frames=16, new_w_h_size=112, n_threads=12):
    with open(ds_list) as f:
        lines = f.readlines()

    block_size = len(lines)/n_threads

    threads=[]

    for i in range(n_threads):
        start = i*block_size
        end = start + block_size
        if i == n_threads-1:
            end = len(lines)
        threads.append(ThreaddatasetMeanCalculator(i, ds_dir, lines[start:end]))
        threads[-1].start()

    sum = np.zeros((new_w_h_size, new_w_h_size, 3))
    std_sum = np.zeros((3))
    count = 0

    for i in range(n_threads):
        threads[i].join()
        sum += threads[i].sum
        std_sum += threads[i].std_sum
        count += threads[i].count

    mean=sum.sum(axis=0).sum(axis=0)/float(count*112*112)
    std_sum=np.sqrt(std_sum/float(count*112*112))
    return mean, std_sum


if __name__ == '__main__':
    mean_dataset_16, std = calculate_dataset_mean("/home/bassel/data/UCF101/frms",
                      "/home/bassel/data/ucf56-120frames/lbl/trainlist01.txt")
    np.save("/home/bassel/data/ucf56-120frames/lbl/crop_mean_stack10_v2.npy", mean_dataset_16)
    np.save("/home/bassel/data/ucf56-120frames/lbl/crop_std_stack10_v2.npy", std)
    print(mean_dataset_16, std)
    import pdb
    pdb.set_trace()
