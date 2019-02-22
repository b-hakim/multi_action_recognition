import os
import shutil
import parse_cls_ind as pci
import glob
import numpy as np


def copy_frames_to_frames_keeping_numbers(cls_ind, from_vds, to_frm_ds):
    cls_ind = pci.parse_cls_ind(cls_ind)
    classes_to_copy = os.listdir(from_vds)

    for class_to_copy in classes_to_copy:
        cls_index = cls_ind[class_to_copy.replace("_2", "")]
        all_cls_vids = glob.glob(os.path.join(to_frm_ds, "{}_*".format(cls_index)))

        cls_vids_numbers = list(map(lambda x: int(x.replace(to_frm_ds+"/", "").replace("{}_".format(cls_index), "")), all_cls_vids))
        cls_vids_numbers.sort()
        if len(cls_vids_numbers) == 0:
            new_vid_number=0
        else:
            new_vid_number = cls_vids_numbers[-1] + 1

        vids_copy = os.listdir(os.path.join(from_vds, class_to_copy))

        for vid_copy in vids_copy:
            files = glob.glob(os.path.join(from_vds, class_to_copy, vid_copy, "*"))

            to_rem_files = list(filter(lambda x: x.find(".jpg") == -1, files))

            for path in to_rem_files:
                os.remove(path)

            continue
            shutil.copytree(os.path.join(from_vds, class_to_copy, vid_copy),
                            os.path.join(to_frm_ds, "{}_{:04}".format(cls_index, new_vid_number)))

            new_vid_number+=1

def missing_class(path):
    vds = os.listdir(path)

    classes = [0]*18

    for vd in vds:
        classes[int(vd.split("_")[0])-1] += 1

    classes = np.array(classes)
    print np.where(classes == 0)[0]+1
    print np.where(classes > 0)[0]+1
    print np.arange(len(np.where(classes > 0)[0]+1))+1

    print classes

if __name__ == '__main__':
    missing_class("/home/bassel/data/oa_kinetics/frms")
# copy_frames_to_frames_keeping_numbers("/home/bassel/data/oa_kinetics/lbls/class_index.txt",
#                                       "/home/bassel/data/oa_kinetics/from frams",
#                                       "/home/bassel/data/oa_kinetics/frms")