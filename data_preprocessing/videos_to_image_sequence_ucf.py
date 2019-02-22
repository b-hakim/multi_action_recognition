import glob
import shutil

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


def select_all_frames_from_video(video_path, output_path):
    video_name = video_path.split("/")[-1]
    save_frame_path = os.path.join(output_path, video_name.split("_")[1], video_name)

    vidcap = cv2.VideoCapture(video_path+".avi")
    success, image = vidcap.read()
    count = 0;

    if not os.path.exists(save_frame_path):
        os.makedirs(save_frame_path)
    elif len(glob.glob(save_frame_path+"/*.jpg")) >= 120:
        return

    while success:
        count += 1
        save_frame_full_path = save_frame_path + '/{:04}'.format(count) + ".jpg"
        cv2.imwrite(save_frame_full_path, image)
        success, image = vidcap.read()
        if count == 120:
            return


def SelectFramesFromSpecificVideosUCF(dataset_dir, output_path, n_threads=12):
    selected_vids = ['Diving/v_Diving_g02_c01', 'Diving/v_Diving_g02_c03', 'Diving/v_Diving_g02_c07', 'Diving/v_Diving_g04_c06', 'Diving/v_Diving_g05_c04', 'Diving/v_Diving_g06_c05', 'Diving/v_Diving_g10_c01', 'Diving/v_Diving_g10_c02', 'Diving/v_Diving_g11_c01', 'Diving/v_Diving_g11_c02', 'Diving/v_Diving_g11_c03', 'Diving/v_Diving_g13_c04', 'Diving/v_Diving_g13_c06', 'Diving/v_Diving_g15_c03', 'Diving/v_Diving_g18_c04', 'Diving/v_Diving_g21_c01', 'Diving/v_Diving_g22_c03', 'Diving/v_Diving_g23_c01', 'Diving/v_Diving_g24_c02', 'Diving/v_Diving_g24_c05', 'Diving/v_Diving_g24_c07', 'Diving/v_Diving_g25_c01', 'HorseRiding/v_HorseRiding_g01_c05', 'HorseRiding/v_HorseRiding_g01_c07', 'HorseRiding/v_HorseRiding_g02_c02', 'HorseRiding/v_HorseRiding_g02_c03', 'HorseRiding/v_HorseRiding_g02_c07', 'HorseRiding/v_HorseRiding_g04_c03', 'HorseRiding/v_HorseRiding_g12_c05', 'HorseRiding/v_HorseRiding_g23_c04', 'PlayingDaf/v_PlayingDaf_g08_c07', 'PoleVault/v_PoleVault_g06_c01', 'PoleVault/v_PoleVault_g08_c04', 'PoleVault/v_PoleVault_g09_c02', 'PoleVault/v_PoleVault_g09_c03', 'PoleVault/v_PoleVault_g13_c03', 'PoleVault/v_PoleVault_g13_c04', 'PoleVault/v_PoleVault_g13_c05', 'PoleVault/v_PoleVault_g17_c04', 'PoleVault/v_PoleVault_g18_c01', 'PoleVault/v_PoleVault_g18_c04', 'PoleVault/v_PoleVault_g18_c06', 'PoleVault/v_PoleVault_g20_c06', 'PoleVault/v_PoleVault_g21_c05', 'PoleVault/v_PoleVault_g22_c01', 'PoleVault/v_PoleVault_g22_c02', 'PoleVault/v_PoleVault_g24_c05', 'PoleVault/v_PoleVault_g24_c06', 'PoleVault/v_PoleVault_g25_c01', 'SoccerJuggling/v_SoccerJuggling_g06_c02', 'SoccerJuggling/v_SoccerJuggling_g10_c04', 'SoccerJuggling/v_SoccerJuggling_g25_c04', 'SoccerJuggling/v_SoccerJuggling_g25_c05', 'SoccerJuggling/v_SoccerJuggling_g25_c06', 'TrampolineJumping/v_TrampolineJumping_g13_c04', 'TrampolineJumping/v_TrampolineJumping_g24_c03', 'TrampolineJumping/v_TrampolineJumping_g25_c01', 'TrampolineJumping/v_TrampolineJumping_g25_c02']
    vids = []

    for vid in selected_vids:
        vids.append(os.path.join(dataset_dir, vid.split('/')[1]))

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
    # SelectFramesFromVideosDataset('/home/bassel/data/office-actions/office_actions_19/short_clips/stabilized_videos/',
    #                       '/media/bassel/My Future/stabilized_frms')
    SelectFramesFromSpecificVideosUCF('/media/bassel/My Future/Study/Masters/datasets/UCF101/videos',
                          '/media/bassel/My Career/datasets/ucf56-4sec-224/frm')
