import Util as utl
import os
import sys
sys.path.insert(0, "../data_preprocessing")
import shuffle_list as shuffle


def isValidFramesStack(clipPath, StartingFrame, NumStackingFrames, step=1):
    last_frame = StartingFrame + (NumStackingFrames-1)*step
    return os.path.exists(os.path.join(clipPath, "{:04}.jpg".format(last_frame)))

def stack_clips(ds_dir, dataset_txtfile, dataset_txtfile_output_path,
                NumStackingFrames = 16, overlapFrames=0, step=1):
    """
    Saves a zero based file with all the starts of each stack
    """
    if not os.path.isdir(utl.get_full_path_for_dir_containing_file(dataset_txtfile_output_path)):
        os.makedirs(utl.get_full_path_for_dir_containing_file(dataset_txtfile_output_path))

    with open(dataset_txtfile) as f:
        lines = f.readlines()

    NewGT = []

    for line in lines:
        l = line.split(',')[0].replace(".mp4", "")
        label = line.split(',')[1]

        # startingFrame = 1

        for startingFrame in range(1, step + 1):
            while True:
                if not isValidFramesStack(os.path.join(ds_dir, l), startingFrame, NumStackingFrames, step):
                    if os.path.exists(os.path.join(ds_dir, l, "{:04}.jpg".format(startingFrame)))\
                            and isValidFramesStack(os.path.join(ds_dir, l), startingFrame, NumStackingFrames/2, step):
                        NewGT += [l + " " + str(startingFrame) + " " + str(step) + " " + str(int(label)) + "\n"]
                    break
                NewGT += [l + " " + str(startingFrame) + " " + str(step) + " " + str(int(label)) + "\n"]
                startingFrame += NumStackingFrames*step

    with open(dataset_txtfile_output_path, 'w') as file_writer:
        file_writer.writelines(NewGT)


if __name__ == '__main__':
    for step in [1, 2, 3, 4, 6, 12]:
        # stack_clips(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        stack_clips("/home/bassel/data/office-actions/office_actions_19/short_clips/resized_frms_224",
                    "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/trainlist.txt",
                    "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/multi_steps/step_"+str(step)+"/trainlist_stacks.txt",
                    NumStackingFrames=10, step=step)

        shuffle.shuffle_list("/home/bassel/data/office-actions/office_actions_19/short_clips/labels/multi_steps/step_"+str(step)+"/trainlist_stacks.txt")

        stack_clips("/home/bassel/data/office-actions/office_actions_19/short_clips/resized_frms_224",
                    "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/testlist.txt",
                    "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/multi_steps/step_"+str(step)+"/testlist_stacks.txt",
                    NumStackingFrames=10, step=step)
