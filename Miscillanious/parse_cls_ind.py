def parse_cls_ind(clsind_path):
    with open(clsind_path) as cls_ind:
        cls_dic = {}

        for line in cls_ind:
            label, cls_name = line.split(',')
            cls_dic[cls_name.strip()] = int(label)
            cls_dic[int(label)] = cls_name.strip()

    return cls_dic