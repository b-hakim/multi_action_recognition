import os

def get_file_name_from_path_without_extention(file_path):
    return file_path[file_path.rfind('/') + 1:file_path.rfind('.')]


def get_file_name_from_path_with_extention(file_path):
    return file_path[file_path.rfind('/') + 1:len(file_path)]

def get_full_path_for_dir_containing_file(file_path):
    ind_end = file_path.rfind('/')
    ind_start = 0
    if ind_end == -1:
        raise Exception("Path does not any dir")
    return file_path[ind_start:ind_end]


def get_direct_folder_containing_file(file_path):
    ind_end = file_path.rfind('/')
    ind_start = file_path[0:ind_end].rfind('/') + 1
    if ind_end == -1:
        raise Exception("Path does not contain folder")
    # print ind_start, ind_end

    return file_path[ind_start:ind_end]

def get_immediate_subdirectories(a_dir, full_path=False):
    if full_path:
        return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]
    else:
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]

def get_frames_count(vid_path):
    import cv2
    cap = cv2.VideoCapture(vid_path)
    cnt = 0
    success = True

    while success:
        success, frm = cap.read()
        if success == False:
            break
        cnt += 1

    return cnt