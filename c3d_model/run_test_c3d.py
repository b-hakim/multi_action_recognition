import numpy as np
from predict_c3d_ucf101 import run_test
from parse_class_index import parse_cls_indx
from evaluate_predictions import evaluate_prediction


ds_dir = "/home/bassel/data/office-actions/office_actions_19/short_clips/stabilized_resized_frms_112"


testing_files_dic = {"all_views":
                         ["/home/bassel/data/office-actions/office_actions_19/short_clips/labels/test_stack_list.txt",
                          "conv3d_deepnetA_sport1m_iter_1900000_TF.model",
                          "../c3d_data_preprocessing/dataset_calculated_mean.npy"],
                 "side_view":
                     ["/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_test_stack_list.txt",
                       "conv3d_deepnetA_sport1m_iter_1900000_TF.model",
                      "../c3d_data_preprocessing/side_action_dataset_calculated_mean.npy"],
                  "front_view":
                      ["/home/bassel/data/office-actions/office_actions_19/short_clips/labels/front_only_test_stack_list.txt",
                       "conv3d_deepnetA_sport1m_iter_1900000_TF.model",
                       "../c3d_data_preprocessing/front_action_dataset_calculated_mean.npy"]}
# testing_files_dic = {"stabilizied side view":
#                          ["/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_test_stack_list.txt",
#                        "stabilized_side_view_c3d_ucf_model-996",
#                           "../c3d_data_preprocessing/stabilized_side_action_dataset_calculated_mean.npy"],
# }
TESTING_BATCH_SIZE=64

cls_indx_path = "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/class_index.txt"
cls_indx = parse_cls_indx(cls_indx_path )


def calculate_cooccurence_matrix(file_path = 'predict_ret.txt'):
    coocurrence_matrix = np.zeros((18,18)).tolist()

    with open(file_path) as predictions:
        for i, l in enumerate(predictions):
            rec = [float(val) for val in l.split(',')]

            coocurrence_matrix[int(rec[0])-1][int(rec[2])-1] += 1

    return coocurrence_matrix


def file_save_coocurrence_matrix(file_path, coocurrence_matrix, cls_indx):
    writer = open(file_path, 'a')

    for action_cls, action_row in enumerate(coocurrence_matrix):
        action_row = [str(x) for x in action_row]
        writer.write("{:20}".format(str(cls_indx[action_cls+1])) + "," + ",".join(action_row))
        writer.write("\n")

    for action_cls, action_row in enumerate(coocurrence_matrix):
        sum = np.array(action_row).sum()
        writer.write(str(cls_indx[action_cls + 1]) + "," +
                     str(100 * action_row[action_cls] / float(sum)) + "%\n")

    writer.close()

if __name__ == '__main__':

    for perspective, (testing_file, model_name, mean_file) in testing_files_dic.items():
            run_test(ds_dir, mean_file, "model/saved_models/"+model_name, testing_file, TESTING_BATCH_SIZE)
            testing_accuracy, _, _ = evaluate_prediction()
            file_path = "sports1M_cooccurence_matrix_{}.csv".format(perspective)

            with open(file_path, 'w') as fw:
                fw.write(str(testing_accuracy)+"\n")

            coocurrence_matrix = calculate_cooccurence_matrix()
            file_save_coocurrence_matrix(file_path, coocurrence_matrix, cls_indx)

