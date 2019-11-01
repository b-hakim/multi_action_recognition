import threading
from vidstab import VidStab
import glob
import os
import shutil


class VideosDatasetStabilizer(threading.Thread):
    def __init__(self, threadID, vids, output_path):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.vids = vids
        self.output_path = output_path

    def run(self):
        for vid in self.vids:
            out_vid = os.path.join(self.output_path, os.path.basename(vid))
            if os.path.isfile(out_vid):
                continue

            try:
                stabilizer = VidStab()
                stabilizer.stabilize(input_path=vid, output_path=out_vid)
            except:
                print ("error in ", vid)
                shutil.copy(vid, out_vid)

def stabilize_dataset(ds_dir, out_ds_dir, nthreads=1):
    vids = glob.glob(os.path.join(ds_dir,"s_*.mp4"))
    workers = []
    batch_size = len(vids)//nthreads

    for i in range(nthreads):
        start = i * batch_size
        end = start + batch_size

        if i == nthreads-1:
            end = len(vids)

        print(start, end)
        workers.append(VideosDatasetStabilizer(i, vids[start:end], out_ds_dir))
        workers[-1].start()


    for worker in workers:
        worker.join()

def copy_remaining_frames(ds_frms_dir, stabilized_frms_dir):
    ds_clips = os.listdir(ds_frms_dir)
    ds_clips = list(filter(lambda x: x.__contains__("s_"), ds_clips))
    stabilized_clips = os.listdir(stabilized_frms_dir)

    ds_clips.sort()
    stabilized_clips.sort()

    for clip, stab_clip in zip(ds_clips, stabilized_clips):
        frames = glob.glob(ds_frms_dir+"/"+clip+"/*.jpg")
        stab_frames = glob.glob(stabilized_frms_dir+"/"+stab_clip+"/*.jpg")

        frames.sort()
        stab_frames.sort()

        if len(frames) > len(stab_frames):
            for i in range(len(stab_frames), len(frames)):
                shutil.copy(frames[i], os.path.join(stabilized_frms_dir, stab_clip, os.path.basename(frames[i])))





if __name__ == '__main__':
    # stabilize_dataset("/home/bassel/data/office-actions/office_actions_19/short_clips/videos",
    #               "/home/bassel/data/office-actions/office_actions_19/short_clips/stabilized_videos")
    #
    copy_remaining_frames("/media/bassel/My Future/frms", "/media/bassel/My Future/stabilized_frms")
