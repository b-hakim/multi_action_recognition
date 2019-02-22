# Training: the rest
# Testing:
# 1, 2, 3, 4, 5, 18, 52, 56, 8, 9, 10, 30
# M6, M24,
# M7,
# F8, F30,
# M12, M33,
# M13, M28,
# F14,
# M15, M27,
# M16, M36,
# F17, F31,
# F21, F32,
# F22, F39
# M29,
# M34,
# F35,
# M36,
# M37,
# F38,

# Males: 10
# Females: 7
unique_videos = [7, 14, 29, 34, 35, 36, 38, 22, 39]

def divide_videos(main_list, out_dir):
    with open(main_list) as fr:
        lines = fr.readlines()

    train_file = out_dir+"/trainlist.txt"
    test_file =  out_dir+"/testlist.txt"

    train_list = []
    test_list = []

    for line in lines:
        clip_name, lbl = line.strip().split(',')
        found = False

        for vid in unique_videos:
            if clip_name.__contains__("_{:03}".format(vid)):
                test_list.append(line.replace(".mp4", ""))
                found=True
                break

        if not found:
            train_list.append(line)

    with open(train_file, 'w') as fw:
        fw.writelines(train_list)

    with open(test_file, 'w') as fw:
        fw.writelines(test_list)


if __name__ == '__main__':
    # hot_encoded = "/home/bassel/data/office-actions/office_actions_19/short_clips_labels/1_hot_encoded.txt"
    hot_encoded = "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/clips_one_action.txt"
    out_dir = "/home/bassel/data/office-actions/office_actions_19/short_clips/labels"

    divide_videos(hot_encoded, out_dir)