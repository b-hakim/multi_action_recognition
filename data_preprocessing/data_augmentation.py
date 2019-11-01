import cv2
import random
import numpy as np
import os


class ImageDataAugmentation:
    @classmethod
    def crop_half_image(cls, img_path, output_path, h=None, w=None):
        img = cv2.imread(img_path)

        if h is None:
            h, w = random.randint(0, 112), random.randint(0, 112)

        img = img[h:h+112, w:w+112, :]
        cv2.imwrite(output_path, img)
        return h, w

    @classmethod
    def add_salf_and_pepper(cls, img_path, output_path):
        img = cv2.imread(img_path)
        salt_vs_pepper = random.randint(1, 5)/10.0
        amount = 0.004

        num_salt = np.ceil(amount * img.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * img.size * (1.0 - salt_vs_pepper))

        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        img[coords[0], coords[1], :] = 0

        cv2.imwrite(output_path, img)

    @classmethod
    def make_light_darker(cls, img_path, output_path, gaussian=None, percentage=None):
        img = cv2.imread(img_path)
        row, col, _ = img.shape

        # Gaussian distribution parameters
        mean = 0
        var = random.randint(1, 9)/10.0#0.1
        sigma = var ** 0.5

        if gaussian is None:
            gaussian = np.random.normal(mean, sigma, (row, col, 1)).astype(np.float32)
            gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
            percentage = random.randint(70, 90)/100.0

        gaussian_img = cv2.addWeighted(gaussian, percentage, img.astype(np.float32), 1-percentage, 0)

        cv2.normalize(gaussian_img, gaussian_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        gaussian_img = gaussian_img.astype(np.uint8)

        # print (gaussian_img-img).sum()
        # cv2.imshow("img", img)
        # cv2.imshow("gaussian", gaussian)
        # cv2.imshow("noisy", gaussian_img)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(output_path, gaussian_img)
        return gaussian, percentage

    @classmethod
    def add_gaussian_noise(cls, img_path, output_path):
        img = cv2.imread(img_path)
        mean = 0
        var = random.randint(10, 75)
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, (224, 224)) # np.zeros((224, 224), np.float32)

        noisy_image = np.zeros(img.shape, np.float32)

        if len(img.shape) == 2:
            noisy_image = img + gaussian
        else:
            noisy_image[:, :, 0] = img[:, :, 0] + gaussian
            noisy_image[:, :, 1] = img[:, :, 1] + gaussian
            noisy_image[:, :, 2] = img[:, :, 2] + gaussian

        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1).astype(np.uint8)
        #
        # cv2.imshow("img", img)
        # cv2.imshow("gaussian", gaussian)
        # cv2.imshow("noisy", noisy_image)
        #
        # cv2.waitKey(0)

        cv2.imwrite(output_path, noisy_image)

    @classmethod
    def add_moving_object(cls, img_path, output_path, x=None, y=None, w=None, h=None):
        img = cv2.imread(img_path)

        if w is None:
            w = random.randint(np.ceil(0.05*img.shape[1]), np.ceil(0.1*img.shape[1]))
            h = random.randint(np.ceil(0.05 * img.shape[0]), np.ceil(0.1 * img.shape[0]))
            x = random.randint(60, 165-w)
            y = random.randint(60, 165-h)

        img[y:y+h, x:x+w, :] = 0

        cv2.imwrite(output_path, img)
        return x, y, w, h


class VideoDataAugmentation:
    @classmethod
    def crop_half_image_dimension(cls, video_frames_dir, output_dir, stack_size):
        frms = os.listdir(video_frames_dir)
        frms.sort()

        for idx, frame_name in enumerate(frms):
            if idx%stack_size == 0:
                x, y = None, None

            frame_path = os.path.join(video_frames_dir, frame_name)
            output_path = os.path.join(output_dir, frame_name)
            x, y = ImageDataAugmentation.crop_half_image(frame_path, output_path, x, y)

    @classmethod
    def add_salt_and_pepper(cls, video_frames_dir, output_dir):
        frms = os.listdir(video_frames_dir)
        frms.sort()

        for frame_name in frms:
            frame_path = os.path.join(video_frames_dir, frame_name)
            output_path = os.path.join(output_dir, frame_name)
            ImageDataAugmentation.add_salf_and_pepper(frame_path, output_path)

    @classmethod
    def make_light_darker(cls, video_frames_dir, output_dir, stack_size):
        frms = os.listdir(video_frames_dir)
        frms.sort()

        for idx, frame_name in enumerate(frms):
            if idx%stack_size == 0:
                gaussian, percentage = None, None

            frame_path = os.path.join(video_frames_dir, frame_name)
            output_path = os.path.join(output_dir, frame_name)
            gaussian, percentage = ImageDataAugmentation.make_light_darker(frame_path, output_path, gaussian, percentage)

    @classmethod
    def add_gaussian_noise(cls, video_frames_dir, output_dir):
        frms = os.listdir(video_frames_dir)
        frms.sort()

        for frame_name in frms:
            frame_path = os.path.join(video_frames_dir, frame_name)
            output_path = os.path.join(output_dir, frame_name)
            ImageDataAugmentation.add_gaussian_noise(frame_path, output_path)

    @classmethod
    def add_moving_object(cls, video_frames_dir, output_dir, stack_size):
        frms = os.listdir(video_frames_dir)
        frms.sort()

        for idx, frame_name in enumerate(frms):
            if idx%stack_size == 0:
                x, y, w, h = [None]*4
                set_direction=None
            frame_path = os.path.join(video_frames_dir, frame_name)
            output_path = os.path.join(output_dir, frame_name)
            x, y, w, h = ImageDataAugmentation.add_moving_object(frame_path, output_path, x, y, w, h)
            if set_direction is None:
                if x >= 110:
                    set_direction=-1
                else:
                    set_direction=1

            move_x = random.randint(0, 1)
            move_y = random.randint(0, 1)

            if move_x:
                x += set_direction*w/3

            if move_y:
                y += set_direction*h/3


def augment_dataset():
    ds_path = "/home/bassel/data/office-actions/office_actions_19/short_clips/resized_frms_224"
    out_dir = "/home/bassel/data/office-actions/office_actions_19/short_clips/resized_frms_224/augmented_from_unstabilized_224"

    train_list = "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/trainlist.txt"
    with open(train_list) as fr:
        lines = fr.readlines()

    final_train_list = []
    stack_size = 16

    for line in lines:
        ## adjust paths and params
        vid_name = line.split(',')[0].replace(".mp4", "")
        vid_path = os.path.join(ds_path, vid_name)
        cls = int(line.split(',')[1].strip())

        ## perform operations
        out_vid_path = os.path.join(out_dir, "crop_img/" + vid_name)
        if not os.path.exists(out_vid_path):
            os.makedirs(out_vid_path)
        VideoDataAugmentation.crop_half_image_dimension(vid_path, out_vid_path, stack_size=stack_size)

        out_vid_path = os.path.join(out_dir, "salf_pepper/" + vid_name)
        if not os.path.exists(out_vid_path):
            os.makedirs(out_vid_path)
        VideoDataAugmentation.add_salt_and_pepper(vid_path, out_vid_path)

        out_vid_path = os.path.join(out_dir, "darker/" + vid_name)
        if not os.path.exists(out_vid_path):
            os.makedirs(out_vid_path)
        VideoDataAugmentation.make_light_darker(vid_path, out_vid_path, stack_size=stack_size)

        # out_vid_path = os.path.join(out_dir, "noisy/"+vid_name)
        # if not os.path.exists(out_vid_path):
        #     os.makedirs(out_vid_path)
        # VideoDataAugmentation.add_gaussian_noise(vid_path, out_vid_path)

        out_vid_path = os.path.join(out_dir, "occlusion/" + vid_name)
        if not os.path.exists(out_vid_path):
            os.makedirs(out_vid_path)
        VideoDataAugmentation.add_moving_object(vid_path, out_vid_path, stack_size=stack_size)

        ## adjust new train list
        final_train_list.append(line)
        final_train_list.append("augmented_from_unstabilized_224/crop_img/" + vid_name + "," + str(cls) + "\n")
        final_train_list.append("augmented_from_unstabilized_224/salf_pepper/" + vid_name + "," + str(cls) + "\n")
        final_train_list.append("augmented_from_unstabilized_224/darker/" + vid_name + "," + str(cls) + "\n")
        # final_train_list.append("augmented_from_unstabilized_224/noisy/"+vid_name+","+str(cls)+"\n")
        final_train_list.append("augmented_from_unstabilized_224/occlusion/" + vid_name + "," + str(cls) + "\n")

    # save new train list
    new_augmented_train_list = "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/augmented_trainlist.txt"
    with open(new_augmented_train_list, 'w') as fw:
        fw.writelines(final_train_list)


def augment_side_front():
    ds="/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_trainlist.txt"
    new_ds="/home/bassel/data/office-actions/office_actions_19/short_clips/labels/augmented_side_only_trainlist.txt"

    final_augmented_train = []

    with open(ds) as fr:
        for line in fr:
            final_augmented_train.append(line)
            final_augmented_train.append("augmented_from_unstabilized_224/crop_img/"+line)
            final_augmented_train.append("augmented_from_unstabilized_224/salf_pepper/"+line)
            final_augmented_train.append("augmented_from_unstabilized_224/darker/"+line)
            final_augmented_train.append("augmented_from_unstabilized_224/occlusion/"+line)

    with open(new_ds, 'w') as fw:
        fw.writelines(final_augmented_train)

    ds="/home/bassel/data/office-actions/office_actions_19/short_clips/labels/front_only_trainlist.txt"
    new_ds="/home/bassel/data/office-actions/office_actions_19/short_clips/labels/augmented_front_only_trainlist.txt"

    final_augmented_train = []

    with open(ds) as fr:
        for line in fr:
            final_augmented_train.append(line)
            final_augmented_train.append("augmented_from_unstabilized_224/crop_img/"+line)
            final_augmented_train.append("augmented_from_unstabilized_224/salf_pepper/"+line)
            final_augmented_train.append("augmented_from_unstabilized_224/darker/"+line)
            final_augmented_train.append("augmented_from_unstabilized_224/occlusion/"+line)

    with open(new_ds, 'w') as fw:
        fw.writelines(final_augmented_train)


if __name__ == '__main__':
   # augment_dataset()
    augment_side_front()