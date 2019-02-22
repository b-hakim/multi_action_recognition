import glob
import os
import shutil


def rename_ds(ds_path, class_ind):
    with open(class_ind) as cls_ind:
        cls_dic = {}

        for line in cls_ind:
            label, cls_name = line.split(',')
            cls_dic[cls_name.strip()] = int(label)
            # cls_dic[int(label)] = cls_name

    for cat in os.listdir(ds_path):
        cat_fullpath = os.path.join(ds_path, cat)
        i=0
        vids = glob.glob(cat_fullpath+"/*.mp4")
        vids.sort()

        for vid in vids:
            vid_fullpath = os.path.join(cat_fullpath, vid)
            vid_new_name = vid_fullpath.replace(os.path.basename(vid_fullpath),
                                                "{}_{:04}.mp4".format(cls_dic[cat], i))
            i+=1
            shutil.move(vid_fullpath, vid_new_name)


if __name__ == '__main__':
    rename_ds("/home/bassel/data/oa_kinetics/videos",
              "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/class_index.txt")