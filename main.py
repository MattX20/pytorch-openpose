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
from pathlib import Path
from typing import List, Union, Optional

BODY_ESTIMATION_MODEL = Path(__file__).parent / 'model'/'body_pose_model.pth'
assert BODY_ESTIMATION_MODEL.exists(), "please download torch models at " \
"https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG"



def get_model(body_estimation=BODY_ESTIMATION_MODEL):
    if isinstance(body_estimation, str) or isinstance(body_estimation, Path):
        body_estimation = Body(BODY_ESTIMATION_MODEL)
    else:
        assert isinstance(body_estimation, Body), f"Wrong model type {type(body_estimation)}"
    return body_estimation

def main(
        image_list: List[Union[Path, str, np.ndarray]],
        vis_dir: Path, save_path: Optional[Path]=None,
        image_names: Optional[str]= None,
        body_estimation: Union[Path, Body]=BODY_ESTIMATION_MODEL
    ) -> np.ndarray:
    """Run Openpose on a set of images.

    Args:
        image_list (List[Union[Path, str, np.ndarray]]):
        a list of L image paths e.g. ['path/to/image1', 'path/to/image2'] 
        or directly a list of numpy arrays.
        vis_dir (Path): output folder path to save debug images
        save_path (Path, optional): Path to save pose dictionaries. Defaults to None.
        body_estimation (Union[Path, Body], optional): Path or loaded body model. Defaults to BODY_ESTIMATION_MODEL.

    Returns:
        np.ndarray: array [L, 18, 3]
    """
    # TODO : process several frames at once using the batch dimension

    if isinstance(vis_dir, str):
        vis_dir = Path(vis_dir)
    num_images = len(image_list)

    # ------------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------------
    body_estimation = get_model(body_estimation)

    # Iterate over input images
    joints_2d = np.zeros((num_images, 18, 3))
    for img_id in range(num_images):
        current_img = image_list[img_id]
        if isinstance(current_img, str) or isinstance(current_img, Path):
            image_path = current_img
            print("Processing {} ...".format(image_path))
            oriImg = cv.imread(image_path) # B,G,R order
        else:
            oriImg = current_img
        

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
        assert vis_dir.exists()
        if image_names is not None:
            im_name = image_names[img_id]
            vis_path = vis_dir/f"{im_name}_pose.png"
        else:
            vis_path = vis_dir/f"{img_id:04d}_pose.png"
        plt.figure()
        # plt.imshow(canvas[:, :, [2, 1, 0]])
        plt.imshow(canvas)
        plt.axis('off')
        plt.savefig(vis_path)
        plt.close()

    # ------------------------------------------------------------
    # Optionally, save joint locations to file
    # ------------------------------------------------------------
    if save_path is not None:
        data_dict = {
            "joint_2d_positions": joints_2d,
            # "image_names": [basename(image_paths[i]) for i in range(num_images)]
        }
        if image_names is not None:
            save_path_current = save_path/f"{image_names[img_id]}.pkl"
        else:
            save_path_current = save_path
            
        with open(save_path_current, 'wb') as f:
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