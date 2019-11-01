import os
import shutil


def adjust_ucf_frms(vd_dir):
    clips = os.listdir(vd_dir)
    clips.sort()

    for clip in clips:
        clp_fp = os.path.join(vd_dir, clip)
        new_path = os.path.join(vd_dir, clip[2:-8], clip)

        if not os.path.isdir(os.path.join(vd_dir, clip[2:-8])):
            os.mkdir(os.path.join(vd_dir, clip[2:-8]))

        shutil.copytree(clp_fp, new_path)
        shutil.rmtree(clp_fp)
adjust_ucf_frms("/home/bassel/data/UCF101/frms")