def transform_test_oa18_to_oa11_kinetics(oa18_test_stacks, oa11_kinetics_test_stacks):
    cls_to_new_class = {2:1, 3:2, 4:3,
                        5:4, 7:5, 9:6,
                        11:7, 13:8, 15:9,
                        17:10, 18:11}

    lines = []

    with open(oa18_test_stacks) as fr:
        for line in fr:
            fname, starting, cls = line.split()
            if int(cls.strip()) in cls_to_new_class:
                cls = str(cls_to_new_class[int(cls.strip())])
                lines.append(fname + " " + starting + " " + cls + "\n")

    with open(oa11_kinetics_test_stacks, "w") as fw:
        fw.writelines(lines)

transform_test_oa18_to_oa11_kinetics("/home/bassel/data/office-actions/office_actions_19/short_clips/labels/test_stack_list.txt",
                                     "/home/bassel/data/oa_kinetics/lbls/oa18_test_stack_mapped_oa11_kinetics.txt")