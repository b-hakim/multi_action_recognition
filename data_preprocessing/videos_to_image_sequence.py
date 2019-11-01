import glob
import threading

import cv2
import Util as utl
import os

class ThreadVideoToImages(threading.Thread):
    def __init__(self, threadID, vids, output_path):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.vids = vids
        self.output_path = output_path

    def run(self):
        for vid in self.vids:
            print (vid)
            select_all_frames_from_video(vid, self.output_path)


def resize_img(img, new_w_h_size=112):
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

    return img


def select_all_frames_from_video(video_path, output_path):
    video_name = utl.get_file_name_from_path_without_extention(video_path)
    save_frame_path = os.path.join(output_path, video_name)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0;

    if not os.path.exists(save_frame_path):
        os.makedirs(save_frame_path)
    # elif glob.glob("*.jpg") > 5:
    #     return

    while success:
        image = resize_img(image)
        count += 1
        save_frame_full_path = save_frame_path + '/{:04}'.format(count) + ".jpg"
        cv2.imwrite(save_frame_full_path, image)
        success, image = vidcap.read()

    os.remove(video_path)


def SelectFramesFromVideosDataset(dataset_dir, output_path, n_threads=12):

    vids = glob.glob(os.path.join(dataset_dir,"*.mp4"))
    if len(vids) == 0:
        vids = glob.glob(os.path.join(dataset_dir,"*.avi"))
    vids.sort()

    block_size = len(vids)/n_threads

    threads = []

    for i in range(n_threads):
        start = block_size*i
        end = start+block_size

        if i == n_threads-1:
            end = len(vids)

        threads.append(ThreadVideoToImages(i, vids[start:end], output_path))
        threads[-1].start()

    for i in range(n_threads):
        threads[i].join()


if __name__ == '__main__':
    # Note it deletes original video
    SelectFramesFromVideosDataset('/home/bassel/data/UCF101/videos',
                          '/home/bassel/data/UCF101/frms')
