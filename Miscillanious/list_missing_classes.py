import glob
import os
import threading
import shutil


def missing_classes_not_generated_in_ucf(ucf56_class_ind, ds_dir):
    with open(ucf56_class_ind) as cls_ind:
        cls_dic = {}

        for line in cls_ind:
            label, cls_name = line.split(' ')
            cls_dic[cls_name.strip()] = int(label)
            # cls_dic[int(label)] = cls_name

    list_generated_classes = os.listdir(ds_dir)
    not_generated_classes = []
    generated_classes = []

    for cls_name in cls_dic.keys():
        if cls_name in list_generated_classes:
            generated_classes.append(cls_name)
        else:
            not_generated_classes.append(cls_name)

    generated_classes.sort()
    not_generated_classes.sort()

    print not_generated_classes, len(not_generated_classes)
    print generated_classes, len(generated_classes)

    return generated_classes, not_generated_classes


def missing_clips_not_generated_in_ucf(ucf56_train_test_dir, ds_dir):
    cls_vids_train_test = []

    with open(os.path.join(ucf56_train_test_dir, "trainlist01.txt")) as trainlist:
        for line in trainlist:
            clip_name, label = line.split(' ')
            cls_vids_train_test.append(clip_name)

    with open(os.path.join(ucf56_train_test_dir, "testlist01.txt")) as testlist:
        for line in testlist:
            clip_name, label = line.split(' ')
            cls_vids_train_test.append(clip_name)

    list_generated_classes = os.listdir(ds_dir)
    list_generated_clips = []

    for cls in list_generated_classes:
        vids = os.listdir(os.path.join(ds_dir, cls))
        vids = list(map(lambda x: os.path.join(cls, x), vids))

        list_generated_clips.extend(vids)

    list_generated_clips.sort()
    cls_vids_train_test.sort()

    print len(list_generated_clips), len(cls_vids_train_test)

    not_generated_clips = []

    for cls_clip in cls_vids_train_test:
        if cls_clip not in list_generated_clips:
            not_generated_clips.append(cls_clip)

    print not_generated_clips
    return not_generated_clips


def clips_less_than_120(ucf56_train_test_dir, ds_dir):
    cls_vids_train_test = []

    with open(os.path.join(ucf56_train_test_dir, "trainlist01.txt")) as trainlist:
        for line in trainlist:
            clip_name, label = line.split(' ')
            cls_vids_train_test.append(clip_name)

    with open(os.path.join(ucf56_train_test_dir, "testlist01.txt")) as testlist:
        for line in testlist:
            clip_name, label = line.split(' ')
            cls_vids_train_test.append(clip_name)

    cls_vids_train_test.sort()

    clips_less_120 = []

    for cls_clip in cls_vids_train_test:
        if len(os.listdir(os.path.join(ds_dir, cls_clip))) < 120:
            clips_less_120.append(cls_clip)

    print clips_less_120
    return clips_less_120


def clips_more_than_120(ucf56_train_test_dir, ds_dir):
    cls_vids_train_test = []

    with open(os.path.join(ucf56_train_test_dir, "trainlist01.txt")) as trainlist:
        for line in trainlist:
            clip_name, label = line.split(' ')
            cls_vids_train_test.append(clip_name)

    with open(os.path.join(ucf56_train_test_dir, "testlist01.txt")) as testlist:
        for line in testlist:
            clip_name, label = line.split(' ')
            cls_vids_train_test.append(clip_name)

    cls_vids_train_test.sort()

    clips_less_120 = []

    for cls_clip in cls_vids_train_test:
        if len(os.listdir(os.path.join(ds_dir, cls_clip))) > 120:
            clips_less_120.append(cls_clip)

    print clips_less_120
    return clips_less_120


def cpy_gen_to_correct_path(gen, old_dir, new_dir):
    for step_i in [1, 2, 3, 4, 6, 12]:
        new_flow_dir = new_dir + "flow/step_{}".format(step_i)
        old_flow_dir = old_dir + "flow/step_{}".format(step_i)
        for cls in gen:
            new_flow_dir += "/"+cls
            old_flow_dir += "/"+cls
            shutil.copytree(old_flow_dir, new_flow_dir)


def copy_120_frames(src_frm_ds, to_frm_ds, clips_to_consider):
    for clip in clips_to_consider:
        src_frame_path = os.path.join(src_frm_ds, clip)
        dst_frame_path = os.path.join(to_frm_ds, clip)

        # if not os.path.isdir(dst_frame_path):
        #     os.makedirs(dst_frame_path)

        shutil.copytree(src_frame_path, dst_frame_path)


class Thread_copy_120_frames (threading.Thread):
    def __init__(self, threadID, src_frm_ds, to_frm_ds, clips_to_consider):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.src_frm_ds=src_frm_ds
        self.to_frm_ds=to_frm_ds
        self.clips_to_consider=clips_to_consider

    def run(self):
        # copy_120_flow(self.src_frm_ds, self.to_frm_ds, self.classes_to_consider)
        copy_120_frames(self.src_frm_ds, self.to_frm_ds, self.clips_to_consider)


def copy_120_flow(src_frm_ds, to_frm_ds, classes_to_consider):
    for step_i in [1, 2, 3, 4, 6, 12]:
        for cat in classes_to_consider:
            print cat, step_i
            videos_in_current_class = os.listdir(os.path.join(src_frm_ds, "step_{}".format(step_i), cat))

            for vid in videos_in_current_class:
                vid_frames = glob.glob(os.path.join(src_frm_ds, "step_{}".format(step_i), cat, vid, "*.jpg"))
                num_frames = len(vid_frames)

                if num_frames >= 120:
                    vid_frames.sort()

                    for i in range(120):
                        vid_frm = vid_frames[i]
                        src_frame_path = vid_frm
                        dst_frame_path = vid_frm.replace(src_frm_ds, to_frm_ds)

                        if os.path.isfile(dst_frame_path):
                            break

                        if not os.path.isdir(os.path.join(to_frm_ds, "step_{}".format(step_i), cat, vid)):
                            os.makedirs(os.path.join(to_frm_ds, "step_{}".format(step_i), cat, vid))

                        shutil.copy(src_frame_path, dst_frame_path)


def parallel_copy_frames(src_frm_ds, to_frm_ds, classes_to_consider, n_threads=12):
    block_size = len(classes_to_consider)/n_threads

    threads=[]

    for i in range(n_threads):
        start = i*block_size
        end = start + block_size

        if i == n_threads-1:
            end = len(classes_to_consider)

        threads.append(Thread_copy_120_frames(i, src_frm_ds, to_frm_ds, classes_to_consider[start:end]))

        threads[-1].start()

    for i in range(n_threads):
        threads[i].join()


def remove_extra_frames(clips_more_120, ds_dir):
    for step in [1, 2, 3, 4, 6, 12]:
        for clip in clips_more_120:
            path = os.path.join(ds_dir, "step_{}".format(step), clip)
            for step_i in range(step):
                last_element = len(glob.glob(os.path.join(path, "{}_flow_x_*.jpg").format(step_i)))
                start = (120/step)

                for i in range(start, last_element+1):
                    fullpath_x = os.path.join(path, "{}_flow_x_{:05}.jpg".format(step_i, i))
                    fullpath_y = os.path.join(path, "{}_flow_y_{:05}.jpg".format(step_i, i))
                    os.remove(fullpath_x)
                    os.remove(fullpath_y)
                    # print fullpath_x

            #
            # for i in range(121, len(os.listdir(path))+1):
            #     os.remove(os.path.join(path, "{:05}.jpg".format(i)))


if __name__ == '__main__':
    # gen, not_gen = missing_classes_not_generated_in_ucf("/home/bassel/data/ucf56-120frames/lbl/class_index.txt",
    #                                      "/media/bassel/My Career/datasets/ucf56-4sec-224_issue_not_correct_list/frm")
    # not_gen_clips = missing_clips_not_generated_in_ucf("/home/bassel/data/ucf56-120frames/lbl/",
    #                                    "/media/bassel/My Career/datasets/ucf56-4sec-224/frm")
    # parallel_copy_frames("/media/bassel/Entertainment/data/UCF101",
    #                      "/media/bassel/My Career/datasets/ucf56-4sec-224/frm",
    #                      not_gen_clips, 12)

    not_gen_clips = clips_more_than_120("/home/bassel/data/ucf56-120frames/lbl/",
                                       "/media/bassel/My Career/datasets/ucf56-4sec-224/frm")

    # clips_more_120 = ['PlayingDaf/v_PlayingDaf_g01_c01', 'PlayingDaf/v_PlayingDaf_g02_c02', 'PlayingDaf/v_PlayingDaf_g06_c04', 'PlayingDaf/v_PlayingDaf_g07_c01', 'PlayingDaf/v_PlayingDaf_g10_c03', 'PlayingDaf/v_PlayingDaf_g14_c06', 'PlayingDaf/v_PlayingDaf_g18_c06', 'PlayingDaf/v_PlayingDaf_g19_c01', 'PlayingDaf/v_PlayingDaf_g20_c04', 'PlayingDaf/v_PlayingDaf_g21_c01', 'PlayingDaf/v_PlayingDaf_g21_c07', 'PlayingDaf/v_PlayingDaf_g23_c01', 'PlayingDaf/v_PlayingDaf_g23_c05', 'PlayingDaf/v_PlayingDaf_g25_c02']
    #
    # remove_extra_frames(clips_more_120, "/media/bassel/My Career/datasets/ucf56-4sec-224/flow")
    # parallel_copy_frames("/media/bassel/My Career/datasets/ucf56-4sec-224_issue_not_correct_list/flow",
    #                      "/media/bassel/My Career/datasets/ucf56-4sec-224/flow",
    #                      gen, 12)
    # cpy_gen_to_correct_path(gen, old_dir="/media/bassel/My Career/datasets/ucf56-4sec-224_issue_not_correct_list/",
    #                         new_dir="/media/bassel/My Career/datasets/ucf56-4sec-224/")