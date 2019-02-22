from predict_c3d_ucf101 import run_test
from train_c3d_ucf101 import run_training
from evaluate_predictions import evaluate_prediction
import shutil
import numpy as np


EPOCHS=1
BATCH_SIZE=32
TESTING_BATCH_SIZE=64

def get_number_steps(_training_file, batch_size, num_epochs):
    with open(_training_file) as fr:
        lines = fr.readlines()

    dataset_size = len(lines)
    num_steps = int(np.ceil((dataset_size/float(batch_size)*num_epochs)))

    return num_steps

ds_dir = "/home/bassel/data/office-actions/office_actions_19/short_clips/stabilized_resized_frms_112"

######################################################################################################################
training_file = "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/37_clp_cls/front_only_train_stack_list.txt"
testing_file = "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/37_clp_cls/front_only_test_stack_list.txt"
# mean_file = "../c3d_data_preprocessing/min_vid_front_action_dataset_calculated_mean.npy"
mean_file = "model/saved_models/tensorflow_model/crop_mean.npy"
visual_dir = "./fix_conv_pretrained_min_vid_front_view_visual_dir"
train_time = run_training(ds_dir, mean_file, visual_dir, EPOCHS, BATCH_SIZE, training_file, testing_file,
                          use_pretrained_model=True,
                          model_filename='model/saved_models/tensorflow_model/sports1m_finetuning_ucf101.model',
                          fix_conv=False)

# from predict_c3d_ucf101 import run_test
num_steps = get_number_steps(training_file, BATCH_SIZE, EPOCHS)
for ext in [".meta", ".index", ".data-00000-of-00001"]:
    old_model_name = "model/c3d_ucf_model-" + str(num_steps - 1)+ ext
    new_model_name = "model/saved_models/fix_conv_pretrained_min_vid_front_view_" + "c3d_ucf_model-" + str(num_steps - 1) + ext
    shutil.move(old_model_name, new_model_name)

new_model_name = "model/saved_models/fix_conv_pretrained_min_vid_front_view_" + "c3d_ucf_model-" + str(num_steps - 1)

run_test(ds_dir, mean_file, new_model_name, training_file, TESTING_BATCH_SIZE)
train_accuracy, _, _ = evaluate_prediction()

run_test(ds_dir, mean_file, new_model_name, testing_file, TESTING_BATCH_SIZE)
testing_accuracy, _, _ = evaluate_prediction()

with open('stats_pretrained.txt', 'a') as f:
    f.write("Training file: " + training_file
            +"\nTesting file: " + testing_file
            # +"\nTraining time: " + str(train_time)
            +"\nTraining accuracy: " + str(train_accuracy)
            +"\nTesting accuracy: " + str(testing_accuracy)
            +"\nModel name: " + new_model_name +"\n")
exit()
########################################################################################################################
training_file = "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/37_clp_cls/side_only_train_stack_list.txt"
testing_file = "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/37_clp_cls/side_only_test_stack_list.txt"
# mean_file = "../c3d_data_preprocessing/min_vid_side_action_dataset_calculated_mean.npy"
mean_file = "model/saved_models/tensorflow_model/crop_mean.npy"
visual_dir = "./fix_conv_pretrained_pretrained_min_vid_side_view_visual_dir"

train_time = run_training(ds_dir, mean_file, visual_dir,
                          EPOCHS, BATCH_SIZE, training_file, testing_file,
                          use_pretrained_model=True,
                          model_filename='model/saved_models/tensorflow_model/sports1m_finetuning_ucf101.model',
                          fix_conv=True)

from predict_c3d_ucf101 import run_test
num_steps = get_number_steps(training_file, BATCH_SIZE, EPOCHS)
for ext in [".meta", ".index", ".data-00000-of-00001"]:
    old_model_name = "model/c3d_ucf_model-" + str(num_steps - 1)+ ext
    new_model_name = "model/saved_models/fix_conv_pretrained_min_vid_side_view_" + "c3d_ucf_model-" + str(num_steps - 1) + ext
    shutil.move(old_model_name, new_model_name)

new_model_name = "model/saved_models/fix_conv_pretrained_min_vid_side_view_" + "c3d_ucf_model-" + str(num_steps - 1)
run_test(ds_dir, mean_file, new_model_name, training_file, TESTING_BATCH_SIZE)
train_accuracy, _, _ = evaluate_prediction()

run_test(ds_dir, mean_file, new_model_name, testing_file, TESTING_BATCH_SIZE)
testing_accuracy, _, _ = evaluate_prediction()

with open('stats_pretrained.txt', 'a') as f:
    f.write("Training file: " + training_file
            +"\nTesting file: " + testing_file
            +"\nTraining time: " + str(train_time)
            +"\nTraining accuracy: " + str(train_accuracy)
            +"\nTesting accuracy: " + str(testing_accuracy)
            +"\nModel name: " + new_model_name)
#######################################################################################################################
training_file = "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/37_clp_cls/train_stack_list.txt"
testing_file = "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/37_clp_cls/test_stack_list.txt"
# mean_file = "../c3d_data_preprocessing/min_vid_all_action_dataset_calculated_mean.npy"
mean_file = "model/saved_models/tensorflow_model/crop_mean.npy"
visual_dir = "./fix_conv_pretrained_min_vid_all_view_visual_dir"

train_time = run_training(ds_dir, mean_file, visual_dir,
                          EPOCHS, BATCH_SIZE, training_file, testing_file,
                          use_pretrained_model=True,
                          model_filename='model/saved_models/tensorflow_model/sports1m_finetuning_ucf101.model',
                          fix_conv = False)

from predict_c3d_ucf101 import run_test

num_steps = get_number_steps(training_file, BATCH_SIZE, EPOCHS)
for ext in [".meta", ".index", ".data-00000-of-00001"]:
    old_model_name = "model/c3d_ucf_model-" + str(num_steps - 1) + ext
    new_model_name = "model/saved_models/fix_conv_pretrained_min_vid_all_views_" + "c3d_ucf_model-" + str(num_steps - 1) + ext
    shutil.move(old_model_name, new_model_name)

new_model_name = "model/saved_models/fix_conv_pretrained_min_vid_all_views_" + "c3d_ucf_model-" + str(num_steps - 1)
run_test(ds_dir, mean_file, new_model_name, training_file, TESTING_BATCH_SIZE)
train_accuracy, _, _ = evaluate_prediction()

run_test(ds_dir, mean_file, new_model_name, testing_file, TESTING_BATCH_SIZE)
testing_accuracy, _, _ = evaluate_prediction()

with open('stats_pretrained.txt', 'a') as f:
    f.write("Training file: " + training_file
            + "\nTesting file: " + testing_file
            + "\nTraining time: " + str(train_time)
            + "\nTraining accuracy: " + str(train_accuracy)
            + "\nTesting accuracy: " + str(testing_accuracy)
            + "\nModel name: " + new_model_name)
########################################################################################################################

