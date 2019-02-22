# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time


def get_frames_data(ds_dir, filename, s_index, num_frames_per_clip=16):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    ret_arr = []

    for parent, dirnames, filenames in os.walk(os.path.join(ds_dir, filename)):
        if(len(filenames)<num_frames_per_clip):
            return [], s_index

        filenames = sorted(filenames)
        # s_index = random.randint(0, len(filenames) - num_frames_per_clip)

        for i in range(s_index, s_index + num_frames_per_clip):
            image_name = os.path.join(ds_dir, filename,str(filenames[i]))
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)
    return ret_arr


np_mean = None


def read_clip_and_label(ds_dir, mean_file, filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112,
                                                shuffle=False, verbose=False, size_adjusted=True):
    global np_mean

    if (start_pos < 0 ):
       print ("error: start_pos is negative and shall not .. terminating")
       raise ValueError("start pos is negative")

    if shuffle:
        print ("shuffle is true .. error")
        raise  ValueError("shuffle is true .. error")

    lines = open(filename,'r')
    read_dirnames = []
    data = []
    labels = []
    batch_index = 0
    next_batch_start = -1
    lines = list(lines)

    if np_mean is None:
        np_mean = np.load(mean_file)\
            .reshape([num_frames_per_clip, crop_size, crop_size, 3])

    # Forcing shuffle, if start_pos is not specified
    if start_pos < 0:
        shuffle = True

    if shuffle:
        video_indices = range(len(lines))
        random.seed(time.time())
        random.shuffle(video_indices)
    # print ("batch size =", batch_size)
    # print ("len(video indices) =", len(video_indices))
    while batch_index < batch_size:

        if not shuffle:
            # Process videos sequentially
            video_indices = range(start_pos, len(lines))

        for index in video_indices:

            line = lines[index].strip('\n').split()
            dirname = line[0]
            # Note that "start-1" as it is an index in a list inside get_frames_data,
            start = int(line[1])-1
            label = int(line[2])-1

            if verbose:
                print("Loading a video clip from {}...".format(dirname))

            tmp_data = get_frames_data(ds_dir, dirname, start, num_frames_per_clip)
            img_datas = [];

            if(len(tmp_data)!=0):
                for j in xrange(len(tmp_data)):
                    img = np.array(Image.fromarray(tmp_data[j].astype(np.uint8)))

                    if not size_adjusted:
                        if(img.shape[1]>img.shape[0]):
                            scale = float(crop_size)/float(img.shape[0])
                            img = cv2.resize(img, (int(img.shape[1] * scale + 1), crop_size)).astype(np.float32)
                        else:
                            scale = float(crop_size)/float(img.shape[1])
                            img = cv2.resize(img, (crop_size, int(img.shape[0]* scale + 1))).astype(np.float32)

                        crop_x = int((img.shape[0] - crop_size)/2)
                        crop_y = int((img.shape[1] - crop_size)/2)

                        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]

                    img_datas.append(img)

                data.append(img_datas)
                labels.append(label)
                batch_index = batch_index + 1
                read_dirnames.append(dirname)

            if (batch_index == batch_size):
                break

        next_batch_start = (start_pos+batch_index) % len(lines)

        if (next_batch_start < start_pos):
            print("batch is switching now to restart from top, batch index, len(video_indices) = ",
                  batch_index, len(video_indices))

        start_pos = (start_pos+batch_index) % len(lines) ## so that when ending loop and not yet finished, restart from 0
        # end while

    # print("batch index =", batch_index)
    # pad (duplicate) data/label if less than batch_size
    valid_len = len(data)
    pad_len = batch_size - valid_len
    # print ("pad len =", pad_len)

    if pad_len:
        for i in range(pad_len):
            data.append(img_datas)
            labels.append(int(label))

    np_arr_data = np.array(data).astype(np.float32)
    np_arr_label = np.array(labels).astype(np.int64)

    return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len

if __name__ == '__main__':
        read_clip_and_label('/home/b.safwat/workspace/datasets/lcw_test_vids/DVRN3529', 16)
