import constants
import motion_cnn
import spatial_cnn
from average_fusion import fusion_accuracry
import os

#
# spatial_cnn.main(ds_path='/home/bassel/data/office-actions/office_actions_19/short_clips/stabilized_resized_frms_224/',
#      trainfile='/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_trainlist.txt',
#      testfile='/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_testlist.txt',
#      prefix="stabilized_side_view")

for step in [3, 4, 6]:
    import motion_cnn
    frame_count_filepath="dataloader/dic/motion_frame_count.pickle"

    if os.path.isfile(frame_count_filepath):
        os.remove(frame_count_filepath)

    ## OA18 dataset
    # motion_cnn.main("/home/bassel/data/office-actions/office_actions_19/short_clips/flow_224/",
    #      trainfile='/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_trainlist.txt',
    #      testfile='/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_testlist.txt',
    #      prefix="step_{}_side_view".format(step),
    #      method=constants.EXPERIMENTS.MULTIPLE_STEPS__CLIPS_START_STEP_END,
    #      step=step)

    ## UCF56
    motion_cnn.main("/media/bassel/My Career/datasets/ucf56-4sec-224/flow/",
                    trainfile='/home/bassel/data/ucf56-120frames/lbl/trainlist01.txt',
                    testfile='/home/bassel/data/ucf56-120frames/lbl/testlist01.txt',
                    prefix="ucf56_step_{}".format(step),
                    method=constants.EXPERIMENTS.MULTIPLE_STEPS__CLIPS_START_STEP_END,
                    step=step,
                    ucf_format=True,
                    is_label_zero_based=True
                    )

    # rgb_preds = 'record/spatial/stabilized_side_view_spatial_video_preds.pickle'
# opf_preds = 'record/motion/stabilized_side_view_motion_video_preds.pickle'
# ds_path = '/home/bassel/data/office-actions/office_actions_19/short_clips/stabilized_resized_frms_224'
# trainfile = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_trainlist.txt'
# testfile = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_testlist.txt'

# fusion_accuracry(rgb_preds, opf_preds, ds_path, trainfile, testfile)