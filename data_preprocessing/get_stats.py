import glob
import matplotlib.pyplot as plt

import numpy as np


def get_actions_frequencies(dataset_file, classIndxPath, specific_clips=[]):
    with open(dataset_file) as fr:
        lines = fr.readlines()

    ## count frequency for each class
    dict = {}

    for line in lines:
        clip_name, aID = line.strip().split(',')

        if specific_clips != []:
            found=False
            for i in specific_clips:
                if clip_name.__contains__("_{:03}".format(i)):
                    found=True
                    break
            if not found:
                continue

        aID = int(aID)

        if not dict.keys().__contains__(aID):
            dict[aID] = 1
        else:
            dict[aID] += 1

    classIndx = {}

    with open(classIndxPath) as fr:
        for line in fr:
            aID, aName = line.strip().split(',')
            aID = int(aID)
            classIndx[aID] = aName
            classIndx[aName] = aID

    keys = list(dict.keys())
    values, keys = zip(*sorted(zip(dict.values(), dict.keys())))

    for v, k in zip(values, keys):
        print (v, classIndx[k])

    sum=0

    for aID in keys:
        print (classIndx[aID], dict[aID])
        sum += dict[aID]

    print sum

    return dict, classIndx


def plot_hist(numbers):
    numbers = np.array(numbers)
    plt.hist(numbers, bins=100)
    plt.title("Length Frequency Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.show()

def get_minmax_action_duration(front_dataset_dir, side_dataset_dir):
    labels = glob.glob(front_dataset_dir+'/*.txt')
    labels.extend(glob.glob(side_dataset_dir+'/*.txt'))
    min = 99999
    max = -1
    lengthes = []

    for lbl in labels:
        with open(lbl) as fr:
            for l in fr:
                aID, s, e = l.split(',')
                e, s = int(e),int(s)
                length = e-s+1

                if length < min:
                    min = length
                elif e-s > max:
                    max = length

                lengthes += [length]

    print("min duration clip: ", min+1, max+1)

    plot_hist(lengthes)


if __name__ == '__main__':
    get_actions_frequencies('/home/bassel/data/office-actions/office_actions_19/short_clips/labels/.txt',
                            '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/class_index.txt')

    # get_actions_frequencies('/home/bassel/data/office-actions/office_actions_19/short_clips/labels/front_only_trainlist.txt',
    #                         '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/class_index.txt')
    #
    # get_actions_frequencies('/home/bassel/data/office-actions/office_actions_19/short_clips/labels/front_only_testlist.txt',
    #                         '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/class_index.txt')

    # dict1, classIndx1 = get_actions_frequencies("/home/bassel/data/office-actions/office_actions_19/short_clips_labels/saved_list",
    #                         "/home/bassel/data/office-actions/office_actions_19/long_videos/class_index.txt",
    #                                             [7, 14, 29, 34, 35, 36, 38, 22, 39])
    #

    #
    # dict, classIndx = get_actions_frequencies("/home/bassel/data/office-actions/office_actions_19/short_clips_labels/saved_list",
    #                                         "/home/bassel/data/office-actions/office_actions_19/long_videos/class_index.txt", [])
    #
    # keys = list(dict.keys())
    # keys.sort()
    #
    # sum=0
    #
    # for aID in keys:
    #     print (classIndx[aID], dict[aID]*0.2, "-->", dict1[aID], dict1[aID] >= int(dict[aID]*0.2))
    #     sum += dict1[aID]
    #
    # print sum
    #
    # get_minmax_action_duration("/home/bassel/data/office-actions/office_actions_19/long_videos/front_view",
    #                            "/home/bassel/data/office-actions/office_actions_19/long_videos/side_view")