import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from glob import glob
from os import makedirs
from os.path import join, exists, basename
import argparse
import copy

from src import model
from src import util
from src.body import Body


def main(image_paths, vis_dir, save_path=None):
    '''
    Run Openpose on a set of images.
    -
    image_paths: a list of image paths, e.g. ['path/to/image1', 'path/to/image2']
    vis_dir: folder path for saving output images
    save_path: the estimated human 2D poses will be saved to file if a valid save_path is provided
    '''

    num_images = len(image_paths)

    # ------------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------------

    body_estimation = Body('model/body_pose_model.pth')

    # Iterate over input images
    joints_2d = np.zeros((num_images, 18, 3))
    for img_id in range(num_images):
        image_path = image_paths[img_id]
        print("Processing {} ...".format(image_path))
        oriImg = cv.imread(image_path) # B,G,R order

        # ------------------------------------------------------------
        # compute subsets
        # ------------------------------------------------------------

        candidate, subset, all_peaks = body_estimation(oriImg)

        # ------------------------------------------------------------
        # Keep the most confident subset
        # ------------------------------------------------------------

        # Initialize the person with no joint and zero confidence
        person = -1*np.ones((20)) # no peaks
        person[-1] = 0. # no detected joints
        person[-2] = 0. # zero score for that person
        c_max = 0.
        if len(subset)>0:
            for i in range(len(subset)):
                if subset[i][-2]>c_max:
                    c_max = subset[i][-2]
                    person = subset[i]

        # Assign the most confident joint peak to missing joints in person
        for i in range(18):
            if person[i]== -1 and len(all_peaks[i])>0:
                # seach the peak with highest score
                joint_peaks = all_peaks[i]
                max_score = 0.
                pid = -1
                for k in range(len(joint_peaks)):
                    if joint_peaks[k][2]>max_score:
                        max_score = joint_peaks[k][2]
                        pid = joint_peaks[k][3]
                person[i] = pid

        for i in range(18):
            pid = person[i].astype(int)
            if pid >= 0:
                for j in range(len(all_peaks[i])):
                    if all_peaks[i][j][3] == pid:
                        joint_position = np.array(all_peaks[i][j][0:3]) # 1d array
                        break

                joints_2d[img_id][i] = joint_position

        # ------------------------------------------------------------
        # Draw estimated joints on input images
        # ------------------------------------------------------------
        
        canvas = copy.deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        # ------------------------------------------------------------
        # Save the image to file
        # ------------------------------------------------------------
        if not exists(vis_dir):
            makedirs(vis_dir)

        vis_path = join(vis_dir, basename(image_path))
        
        plt.figure()
        plt.imshow(canvas[:, :, [2, 1, 0]])
        plt.axis('off')
        plt.savefig(vis_path)
        plt.close()

    # ------------------------------------------------------------
    # Optionally, save joint locations to file
    # ------------------------------------------------------------
    if save_path is not None:
        data_dict = {
            "joint_2d_positions": joints_2d,
            "image_names": [basename(image_paths[i]) for i in range(num_images)]
        }
        with open(save_path, 'wb') as f:
            pk.dump(data_dict, f)

    return joints_2d

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Run Openpose on all images within a folder.')
    parser.add_argument(
        "input_dir", help="Path to a folder containing images.")
    parser.add_argument(
        "vis_dir", help="Path to another folder for saving output visualization images")
    parser.add_argument(
        "save_path", help="Path for saving output joint locations.")

    args = parser.parse_args()
    input_dir = args.input_dir
    vis_dir = args.vis_dir
    save_path = args.save_path

    # Retrieve image paths from the input image folder
    image_extensions = ("jpg", "png")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(sorted(glob(join(input_dir, "*.{0:s}".format(ext)))))

    main(image_paths, vis_dir, save_path)