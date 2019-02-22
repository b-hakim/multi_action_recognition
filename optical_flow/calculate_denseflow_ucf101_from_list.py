import math

import cv2
import fnmatch
import multiprocessing
import os
import glob
import time
from math import ceil
import shutil


def initialize_dirs(ds_videos_paths, ds_dir, out_optical_flow_dataset):
    for video_path in ds_videos_paths:
        # calculate for each video the optical flow output path

        for step in [1, 2, 3, 4, 6, 12]:
            s = os.path.join(out_optical_flow_dataset, "step_{}".format(str(step)))
            output_denseflow = video_path.replace(ds_dir, s)

            if not os.path.isdir(output_denseflow):
                os.makedirs(output_denseflow)


def list_all_videos_ucf101(ds_dir, out_optical_flow_dataset):
    not_generated_list = ['ApplyEyeMakeup', 'BandMarching', 'Billiards', 'BlowDryHair', 'BrushingTeeth', 'Diving',
                          'Drumming', 'HammerThrow', 'HorseRiding', 'Kayaking', 'Nunchucks', 'PlayingFlute',
                          'PlayingTabla', 'PoleVault', 'PommelHorse', 'Rafting', 'RockClimbingIndoor', 'ShavingBeard',
                          'SkyDiving', 'SoccerJuggling', 'Surfing', 'TableTennisShot', 'TrampolineJumping', 'Typing',
                          'WritingOnBoard', 'YoYo']

    # list all videos
    ds_videos_paths = []

    for cat in not_generated_list:
        path_to_category = os.path.join(ds_dir, cat)

        for clip in os.listdir(path_to_category):
            clip_path = os.path.join(path_to_category, clip)

            if len(glob.glob(os.path.join(clip_path, "*.jpg"))) < 120:
                continue

            cont = True

            for step in [1, 2, 3, 4, 6, 12]:
                output_clip_path = clip_path.replace(ds_dir,
                                                     os.path.join(out_optical_flow_dataset, "step_{}".format(str(step))))

                if not os.path.isdir(output_clip_path) or len(os.listdir(output_clip_path)) < ((120.0/step)-1)*step*2:
                    cont = False
                    break

            if cont:
                print "skipping " + str(output_clip_path)
                continue

            ds_videos_paths.append(clip_path)

        # ucf_videos_paths.extend(list(map(lambda x: os.path.join(UCF101_dataset, cat, x), os.listdir(os.path.join(UCF101_dataset, cat)))))

    # for root, dirnames, filenames in os.walk(UCF101_dataset):
        # for filename in fnmatch.filter(filenames, '*.avi'):
                # ucf_videos_paths.append(os.path.join(root, filename))
    ds_videos_paths.sort()
    return ds_videos_paths


def calculate_colored_denseflow_video(video_path, ds_dir, out_optical_flow_dataset, of_gpu_path):
    # calculate for each video the optical flow output path
    cmd = of_gpu_path +" -f='{}' -x='{}/{}_flow_x' -y='{}/{}_flow_y' " \
          "-i='{}/image' -b=20 -t=1 -d=0 -s={} -o=dir -sf={}"

    for step in [1, 2, 3, 4, 6, 12]:
        output_denseflow = video_path.replace(ds_dir,
                                              os.path.join(out_optical_flow_dataset, "step_{}".format(str(step))))


        # print "\tstep " + str(step)

        for start in range(step):

            if len(glob.glob(output_denseflow + "/{}_*.jpg".format(start))) > 15:
                continue

            # Run the cmd command to calculate the optical flow
            command = cmd.format(video_path, output_denseflow, start, output_denseflow, start,
                                 output_denseflow, step, start)
            try:
                os.system(command)
            except:
                print('FATAL [default] CRASH HANDLED; Application has crashed due to [SIGSEGV] signal')


def calculate_optical_flow_UCF101(ds_dir, out_optical_flow_dataset, NUM_PARALLEL_JOBS, of_gpu_path):
    ds_videos_paths = list_all_videos_ucf101(ds_dir, out_optical_flow_dataset)
    ds_videos_paths.sort()
    initialize_dirs(ds_videos_paths, ds_dir, out_optical_flow_dataset)

    batch_num = int(ceil(len(ds_videos_paths)/float(NUM_PARALLEL_JOBS)))

    if len(ds_videos_paths) < NUM_PARALLEL_JOBS:
        NUM_PARALLEL_JOBS = len(ds_videos_paths)

    print "number of batches: {}".format(str(batch_num))

    for i in range(batch_num):
        t = time.time()
        start = i * NUM_PARALLEL_JOBS
        end = start + NUM_PARALLEL_JOBS

        if i == batch_num-1:
            end = len(ds_videos_paths)

        jobs = []

        for j in range(start, end):
            p = multiprocessing.Process(target=calculate_colored_denseflow_video, args=(ds_videos_paths[j], ds_dir,
                                                                                        out_optical_flow_dataset,
                                                                                        of_gpu_path))
            jobs.append(p)
            p.start()

        print ("{} jobs started".format(NUM_PARALLEL_JOBS))

        for j in jobs:
            j.join()

        print ("{} jobs joined".format(NUM_PARALLEL_JOBS))

        print "step {}: {} seconds".format(str(i), time.time()-t)

    # p.map(calculate_colored_denseflow_video, ucf_videos_paths)

if __name__ == '__main__':
    of_gpu_path = "/home/bassel/eclipse-workspace/dense_flow/build/extract_gpu"
    # ds_dir = "/media/bassel/My Career/datasets/ucf56-4sec-224/frm"
    ds_dir = "/media/bassel/My Career/datasets/ucf56-4sec-224/frm"
    OUTPUT_optical_flow_dataset = "/media/bassel/My Career/datasets/ucf56-4sec-224/flow"
    NUM_PARALLEL_JOBS = 16

    calculate_optical_flow_UCF101(ds_dir, OUTPUT_optical_flow_dataset, NUM_PARALLEL_JOBS, of_gpu_path)