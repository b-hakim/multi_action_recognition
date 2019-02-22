import glob
import os

from shuffle_list import shuffle_list


def generate_list_for_vids_oa_format(ds_path, listing_path):
    vids = os.listdir(ds_path)
    cls_to_new_class = {2:1, 3:2, 4:3,
                        5:4, 7:5, 9:6,
                        11:7, 13:8, 15:9,
                        17:10, 18:11}
    lines = []

    for vid in vids:
        cls = cls_to_new_class[int(vid.split("_")[0])]
        lines.append(vid + "," + str(cls) + "\n")

    with open(listing_path, "w") as fw:
        fw.writelines(lines)


if __name__ == '__main__':
    generate_list_for_vids_oa_format("/home/bassel/data/oa_kinetics/frms",
                                     "/home/bassel/data/oa_kinetics/lbls/action_listing.txt")
    shuffle_list("/home/bassel/data/oa_kinetics/lbls/action_listing.txt")