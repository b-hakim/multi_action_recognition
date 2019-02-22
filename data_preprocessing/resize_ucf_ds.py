import glob
import os
import threading
import cv2


class ThreadDatasetResize (threading.Thread):
    def __init__(self, threadID, out_dir, frames_paths, new_w_h_size=112):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.frames_paths = frames_paths
        self.out_dir = out_dir
        self.new_w_h_size = new_w_h_size

    def run(self):
        thread_resize_dataset(self.frames_paths, self.out_dir, self.new_w_h_size)


def thread_resize_dataset(frames_paths, out_dir, new_w_h_size=112):

    for frame_path in frames_paths:
        new_frame_dir = os.path.join(out_dir, frame_path.split('/')[-2])
        new_frame_path = os.path.join(new_frame_dir+"/",frame_path.split('/')[-1])

        if os.path.isfile(new_frame_path):
            continue

        if not os.path.isdir(new_frame_dir):
            os.mkdir(new_frame_dir)

        img = cv2.imread(frame_path)
        height, width, _ = img.shape

        if (width > height):
            scale = float(new_w_h_size) / float(height)
        else:
            scale = float(new_w_h_size) / float(width)

        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        height, width, _ = img.shape

        crop_y = int((height - new_w_h_size) / 2)
        crop_x = int((width - new_w_h_size) / 2)
        img = img[crop_y:crop_y + new_w_h_size, crop_x:crop_x + new_w_h_size, :]

        cv2.imwrite(new_frame_path, img)


def resize_dataset(frms_dir, out_dir, new_w_h_size=112, n_threads=12):
    frames = glob.glob(frms_dir+"/*/*/*.jpg")
    frames.sort()

    block_size = len(frames)/n_threads

    threads=[]

    for i in range(n_threads):
        start = i*block_size
        end = start + block_size

        if i == n_threads-1:
            end = len(frames)

        threads.append(ThreadDatasetResize(i, out_dir, frames[start:end], new_w_h_size))
        threads[-1].start()

    for i in range(n_threads):
        threads[i].join()


resize_dataset("/media/bassel/Entertainment/data/ucf56-4sec/frm",#/home/bassel/data/office-actions/office_actions_19/short_clips/frms
               "/media/bassel/My Career/datasets/ucf56-224/frm",
               224)