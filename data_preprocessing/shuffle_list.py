import numpy as np


def shuffle_list(list_path):
    with open(list_path) as list:
        l = list.readlines()

    # np.random.seed(48)
    np.random.shuffle(l)

    with open(list_path, 'w') as list_writer:
        list_writer.writelines(l)

if __name__ == '__main__':
    shuffle_list("/home/bassel/data/office-actions/office_actions_19/short_clips/labels/augmented_train_stack_list.txt")