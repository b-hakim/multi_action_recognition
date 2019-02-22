def filter_list(clip_actions_lst, clip_actions_new_lst):
    lines = []

    with open(clip_actions_lst) as fr:
        for line in fr:

            if len(line.split(';')) > 1:
                continue

            line = line.split()[0] + "\n"

            lines.append(line)

    with open(clip_actions_new_lst, 'w') as fw:
        fw.writelines(lines)



if __name__ == '__main__':
    filter_list("/home/bassel/data/office-actions/office_actions_19/short_clips/labels/clip_actions.txt",
                "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/clips_one_action.txt")