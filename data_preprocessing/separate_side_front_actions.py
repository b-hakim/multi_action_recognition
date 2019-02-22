def separate_front_and_side(in_file, side_out_file, front_out_file):
    with open(in_file) as fr:
        f = []
        s = []
        for line in fr:
            if line.__contains__("f_"): f.append(line)
            if line.__contains__("s_"): s.append(line)

    with open(side_out_file, "w") as fw:
        fw.writelines(s)

    with open(front_out_file, "w") as fw:
        fw.writelines(f)

# separate_front_and_side("/home/bassel/data/office-actions/office_actions_19/short_clips/labels/train_stack_list.txt",
#                 "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_train_stack_list.txt",
#                 "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/front_only_train_stack_list.txt")
#
# separate_front_and_side("/home/bassel/data/office-actions/office_actions_19/short_clips/labels/test_stack_list.txt",
#                 "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_test_stack_list.txt",
#                 "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/front_only_test_stack_list.txt")

separate_front_and_side("/home/bassel/data/office-actions/office_actions_19/short_clips/labels/trainlist.txt",
                "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_trainlist.txt",
                "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/front_only_trainlist.txt")

separate_front_and_side("/home/bassel/data/office-actions/office_actions_19/short_clips/labels/testlist.txt",
                "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_testlist.txt",
                "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/front_only_testlist.txt")