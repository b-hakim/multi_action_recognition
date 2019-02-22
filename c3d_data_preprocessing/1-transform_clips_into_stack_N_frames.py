import Util as utl
import os


def isValidFramesStack(clipPath, StartingFrame, NumStackingFrames):
    return os.path.exists(os.path.join(clipPath, "{:04}.jpg".format(StartingFrame+NumStackingFrames-1)))

def stack_clips(ds_dir, dataset_txtfile, dataset_txtfile_output_path,
                NumStackingFrames = 16, overlapFrames=0):
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

        startingFrame = 1

        while isValidFramesStack(os.path.join(ds_dir, l), startingFrame, NumStackingFrames):
            NewGT += [l +" " + str(startingFrame) + " "+ str(int(label)) + "\n"]
            startingFrame += NumStackingFrames-overlapFrames

    with open(dataset_txtfile_output_path, 'w') as file_writer:
        file_writer.writelines(NewGT)

if __name__ == '__main__':
    import sys
    # stack_clips(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    stack_clips("/home/bassel/data/oa_kinetics/frms",
                "/home/bassel/data/oa_kinetics/lbls/action_listing.txt",
                "/home/bassel/data/oa_kinetics/lbls/actions_stack_list.txt")