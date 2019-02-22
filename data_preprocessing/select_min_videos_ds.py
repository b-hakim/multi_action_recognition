import random


def select_min_num_videos_ds(ds_file, out_ds_file, min_number_clips_cls=37):
    with open(ds_file) as fr:
        ds_list = fr.readlines()

    dict = {}

    for line in ds_list:
        clip = line.split(',')[0]
        cls = int(line.split(',')[1])

        if dict.has_key(cls):
            dict[cls].append(clip)
        else:
            dict[cls] = [clip]

    lines = []

    for key, value in dict.items():
        random.shuffle(value)
        value = value[0:min_number_clips_cls]

        for val in value:
            lines.append(val+","+str(key)+"\n")

    with open(out_ds_file, "wb") as fw:
        fw.writelines(lines)

if __name__ == '__main__':
    select_min_num_videos_ds("/home/bassel/data/office-actions/office_actions_19/short_clips/labels/front_only_trainlist.txt",
                             "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/37_clp_cls/front_only_trainlist.txt",
                             15)