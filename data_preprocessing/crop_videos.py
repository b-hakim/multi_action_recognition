import threading
import time
import cv2
import glob
import os


def divide_videos(vid_dir, out_dir, mode='a', n_threads=12):
    vds = glob.glob(vid_dir + "/*.mp4")
    vds.sort()

    lbls = glob.glob(vid_dir + "/*.txt")
    lbls.sort()

    vds = vds[0:len(lbls)]
    final_labeling = []

    batch_size = len(vds)//n_threads

    helper_threads = []

    for batch_idx in range(n_threads):
        start_idx = batch_idx*batch_size
        end_idx = start_idx + batch_size

        if batch_idx == n_threads-1:
            end_idx = len(vds)

        helper_threads.append(SplitVideos(batch_idx, vds[start_idx:end_idx], lbls[start_idx:end_idx], out_dir))
        helper_threads[-1].start()

    for helper_thread in helper_threads:
        helper_thread.join()
        final_labeling.extend(helper_thread.final_labeling)

    with open(out_dir+"/saved_list", mode) as fw:
        fw.writelines(final_labeling)


def divide_video(vid, label, out_dir):
    cap = cv2.VideoCapture(vid)
    h, w, _ = cap.read()[1].shape
    output_vid_name = out_dir + "/" + cv2.os.path.basename(vid)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(vid)

    with open(label) as fr:
        lines = fr.readlines()

    final_labeling = []

    for i, line in enumerate(lines):
        clip_name = output_vid_name.replace(".mp4", "_{:02}.mp4".format(i))
        out = cv2.VideoWriter(clip_name, fourcc, 30.0, (w, h))

        if line.strip() == "" or line is None or line.replace(" ", "") == "":
            continue

        action_id, starting_frame, ending_frame = line.strip().split(",")

        final_labeling.append(os.path.basename(clip_name) + " " + action_id + "\n")

        starting_frame = int(starting_frame)
        ending_frame = int(ending_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)

        for ind in range(starting_frame, ending_frame + 1):
            out.write(cap.read()[1])

        out.release()
    cap.release()

    return final_labeling


class SplitVideos (threading.Thread):
    def __init__(self, threadID, lst_video_path, lst_video_label_path, out_dir):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.lst_video_path= lst_video_path
        self.lst_video_label_path = lst_video_label_path
        self.out_dir = out_dir

    def run(self):
        self.final_labeling = []

        for video_path, video_label_path in zip(self.lst_video_path, self.lst_video_label_path):
            start = time.time()
            # if not video_path.__contains__("f_008.mp4"):
            #     continue
            self.final_labeling.extend(divide_video(video_path, video_label_path, self.out_dir))
            print("time taken: " + str(time.time() - start) + " seconds")


def sort_labels(dir):
    lbls = glob.glob(dir + "/*.txt")

    for lbl in lbls:
        with open(lbl) as fr:
            lines = fr.readlines()

        lines.sort(key=lambda x: int(x.split(',')[1]))

        with open(lbl, 'w') as fw:
            fw.writelines(lines)


if __name__ == '__main__':
    vid_dir = "/home/bassel/data/office-actions/office_actions_19/long_videos/front_view"
    out_dir = "/home/bassel/data/office-actions/office_actions_19/short_clips/videos"
    divide_videos(vid_dir, out_dir)

    vid_dir = "/home/bassel/data/office-actions/office_actions_19/long_videos/side_view"
    out_dir = "/home/bassel/data/office-actions/office_actions_19/short_clips/videos"
    divide_videos(vid_dir, out_dir)

    sort_labels(vid_dir)