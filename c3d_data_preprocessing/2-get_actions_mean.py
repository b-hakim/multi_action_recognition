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
    def run(self):
        self.sum, self.count = calculate_thread_dataset_mean(self.ds_dir, self.ds_train_lst)


def calculate_thread_dataset_mean(ds_dir, ds_train_lst, num_frames=16, new_w_h_size=112):
    sum = np.zeros((num_frames, new_w_h_size, new_w_h_size, 3))
    count = 0

    for line in ds_train_lst:
        vid_path = line.strip().split()[0].replace(".mp4", "")
        start_pos = int(line.split()[1])
        lbl = int(line.split()[2])

        stack_frames = []

        for i in range(start_pos, start_pos+num_frames):
            img = cv2.imread(os.path.join(ds_dir, vid_path, "{:04}.jpg".format(i)))
            # img = cv2.resize(img, (new_w_h_size, new_w_h_size))
            if img is None:
                print os.path.join(ds_dir, vid_path, "{:04}.jpg".format(i))
            height, width, _ = img.shape

            if (width > height):
                scale = float(new_w_h_size) / float(height)
            else:
                scale = float(new_w_h_size) / float(width)

            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

            height, width, _ = img.shape

            crop_y = int((height - new_w_h_size) / 2)
            crop_x = int((width - new_w_h_size) / 2)

            stack_frames.append(img[crop_y:crop_y + new_w_h_size, crop_x:crop_x + new_w_h_size, :])

            # stack_frames.append(img)

        stack_frames = np.array(stack_frames)
        sum += stack_frames
        count += 1
    # mean/=float(count)
    # print mean
    return sum, count


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

    sum = np.zeros((num_frames, new_w_h_size, new_w_h_size, 3))
    count = 0

    for i in range(n_threads):
        threads[i].join()
        sum += threads[i].sum
        count += threads[i].count

    mean=sum/float(count)
    return mean


if __name__ == '__main__':
    mean_dataset_16 = calculate_dataset_mean("/home/bassel/data/oa_kinetics/frms",
                      "/home/bassel/data/oa_kinetics/lbls/actions_stack_list.txt")
    np.save("oa_kinetics_calculated_mean.npy", mean_dataset_16)

    print(np.load("oa_kinetics_calculated_mean.npy").sum())