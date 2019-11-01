import numpy as np


def save_list_to_file(z_list, z_file):
    with open(z_file, 'w') as fw:
        fw.writelines(z_list)


def random_split_train_test(train_file, out_train_file, out_test_file, train_percentage=0.8):
    with open(train_file) as fr:
        lines = fr.readlines()

    np.random.shuffle(lines)

    train_data, test_data = lines[0:int(train_percentage*len(lines))], lines[int(train_percentage*len(lines)):]

    save_list_to_file(train_data, out_train_file)
    save_list_to_file(test_data, out_test_file)

random_split_train_test("/home/bassel/data/oa_kinetics/lbls/actions_stack_list.txt",
                        "/home/bassel/data/oa_kinetics/lbls/action_train_stacks_list.txt",
                        "/home/bassel/data/oa_kinetics/lbls/action_test_stacks_list.txt")