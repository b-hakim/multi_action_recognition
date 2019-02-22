import glob
import os


def intersections(a1_start, a1_end, a2_start, a2_end, relative_to_a1=False):
    intersection_len = a1_end-a2_start + 1

    if intersection_len <= 0:
        return 0

    len_a1 = a1_end - a1_start + 1
    len_a2 = a2_end - a2_start + 1

    if relative_to_a1:
        return float(intersection_len) / len_a1

    return float(intersection_len) / len_a2

def adjust_one_hot_encoding_labeling(dataset_dir, hot_encoding_path, n_actions, all_actions_threshold=0.3,
                                     main_action_threshold=0.25, mode='a'):
    labels = glob.glob(dataset_dir+"/*.txt")
    labels.sort()
    final_labels = []

    for vid_label in labels:
        lbl_name = os.path.basename(vid_label)

        with open(vid_label) as fr:
            lines = fr.readlines()

        for i in range(len(lines)):
            if lines[i].strip() == "":
                continue

            a1_id, a1_start, a1_end = lines[i].split(",")
            a1_id, a1_start, a1_end = int(a1_id), int(a1_start), int(a1_end)

            # a1_labels = [0]*n_actions
            # a1_labels[a1_id-1] = 1
            a1_labels = str(a1_id) + " " + str(a1_start) + " " + str(a1_end)

            for j in range(len(lines)):
                clip_name = lbl_name.replace(".txt", "_{:02}.mp4".format(i))

                if i == j:
                    continue

                if lines[j].strip() == "":
                    continue

                a2_id, a2_start, a2_end = lines[j].split(",")
                a2_id, a2_start, a2_end = int(a2_id), int(a2_start), int(a2_end)

                start = a1_start if a1_start < a2_start else a2_start
                end = a1_end if a2_end > a1_end else a2_end

                if start == a1_start:
                    if intersections(a1_start, a1_end, a2_start, a2_end) >= all_actions_threshold\
                            and intersections(a1_start, a1_end, a2_start, a2_end, True) >=  main_action_threshold:
                        # a1_labels[a2_id-1] = 1
                        a1_labels += "; " + str(a2_id) + " " + str(start-a1_start) + " " + str(end-a1_start)
                else:
                    if intersections(a2_start, a2_end, a1_start, a1_end, True) >= all_actions_threshold\
                            and intersections(a2_start, a2_end, a1_start, a1_end) >= main_action_threshold:
                        # a1_labels[a2_id-1] = 1
                        a1_labels +=  "; " + str(a2_id) + " " + str(start-a2_start) + " " + str(end-a2_start)

            final_labels.append(clip_name+ "," + a1_labels + "\n")

    with open(hot_encoding_path, mode) as fw:
        fw.writelines(final_labels)



if __name__ == '__main__':
    adjust_one_hot_encoding_labeling("/home/bassel/data/office-actions/office_actions_19/long_videos/front_view",
                                     "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/1_hot_encoded.txt",
                                     18, mode='w')

    adjust_one_hot_encoding_labeling("/home/bassel/data/office-actions/office_actions_19/long_videos/side_view",
                                     "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/1_hot_encoded.txt",
                                     18, mode='a')