import os
import glob
import pickle
from  list_missing_classes import missing_classes_not_generated_in_ucf


def count_num_videos_per_class(ds_dir, num_frames_above_or_equal, num_videos_above_or_equal):
    categories = os.listdir(ds_dir)
    dic_class_action = {}
    dic_class_action_criteria = {}

    for cat in categories:
        videos_in_current_class = os.listdir(os.path.join(ds_dir, cat))

        dic_class_action[cat] = len(videos_in_current_class)

        count_vds_frames_above_threshold = 0

        for vid in videos_in_current_class:
            if len(glob.glob(os.path.join(ds_dir, cat, vid, "*.jpg"))) >= num_frames_above_or_equal:
                count_vds_frames_above_threshold += 1

        if count_vds_frames_above_threshold >= num_videos_above_or_equal:
            dic_class_action_criteria[cat] = count_vds_frames_above_threshold

    # print ("Dataset class: #videos, num_classes: {}\n".format(len(dic_class_action.keys())))
    # for key, value in dic_class_action.items():
    #     print key, ":", value, "\n"

    print "Dataset class: #videos above or equal: {}, num_classes: {}\n".format(num_frames_above_or_equal,
                                                                                 len(dic_class_action_criteria))

    for key, value in dic_class_action_criteria.items():
        print key, ":", value


def count_num_videos_in_given_class(ds_dir, cls_list):
    categories = os.listdir(ds_dir)
    dic_class_action = {}
    min, max = 1000, -1

    for cat in categories:
        if cat not in cls_list:
            continue

        videos_in_current_class = os.listdir(os.path.join(ds_dir, cat))
        min_num_frames = 1000
        count_videos_greater_equal_120=0

        for vid in videos_in_current_class:
            num_frames = len(glob.glob(os.path.join(ds_dir, cat, vid, "*.jpg")))

            if num_frames >= 120:
                count_videos_greater_equal_120 += 1

                if num_frames < min_num_frames:
                    min_num_frames = num_frames

        dic_class_action[cat] = [count_videos_greater_equal_120, min_num_frames]

        if min > count_videos_greater_equal_120:
            min = count_videos_greater_equal_120

        if max < count_videos_greater_equal_120:
            max = count_videos_greater_equal_120


    print ("Dataset class: #videos, min: {}, max: {}\n".format(min, max))

    for key, value in dic_class_action.items():
        print key, ":", value

if __name__ == '__main__':
    ds_dir="/media/bassel/Entertainment/data/UCF101"
    # count_num_videos_per_class(ds_dir, 120, 72)

    # cls_list = "/media/bassel/Entertainment/data/ucf56-120frames/lbl/out.txt"
    #
    # with open(cls_list) as fr:
    #     cls_list = fr.readline().strip().split(',')
    #
    # for i in range(len(cls_list)):
    #     cls_list[i] = cls_list[i].strip()
    #
    # count_num_videos_in_given_class(ds_dir, cls_list[:-1])
    missing_classes_not_generated_in_ucf("/media/bassel/Entertainment/data/ucf56-120frames/lbl/classInd.txt",
                                         )
