def parse_cls_indx(path):

    with open(path) as fr:
        lines = fr.readlines()

    dic = {}

    for l in lines:
        cls, name = l.split(',')
        dic[int(cls)] = name.strip()
        dic[name.strip()] = int(cls)

    return dic
