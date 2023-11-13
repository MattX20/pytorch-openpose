import argparse
import pickle as pk
from os import makedirs
from os.path import join, exists, dirname

from main import main


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run Openpose on a image folder with still images and image sequences,"
        "e.g., a folder with image_1.jpg, image_1.png, video/frame_{1-100}.png ..."
        "The image_folder is supposed to have a data_info.pkl file.")
    parser.add_argument(
        'image_folder', type=str, metavar='DIR',
        help="Path to the image folder.")
    parser.add_argument(
        'vis_folder', type=str, metavar='DIR',
        help="Location of a folder for saving visualization images.")
    parser.add_argument(
        'save_path', type=str, metavar='DIR',
        help='Location of the output 2D joint locations.')
    parser.add_argument(
        "--item-start", type=int, default=1,
        help="Run program from item #.")
    parser.add_argument(
        "--item-end", type=int, default=-1,
        help="Stop program at item # (the very last item by default).")
    parser.add_argument(
        "--save-after-each-iteration", default=False, action="store_true",
        help="Save the output data_dict after each iteration over (video) items."
        "Otherwise the data_dict will not be saved until all items are computed.")

    args = parser.parse_args()
    image_folder = args.image_folder
    vis_folder = args.vis_folder
    save_path = args.save_path
    item_start = args.item_start
    item_end = args.item_end
    save_after_each_iteration = args.save_after_each_iteration

    # ------------------------------------------------------------
    # Load useful information

    print("Loading image folder info from {0:s} ...".format(
        join(image_folder, "data_info.pkl")))
    with open(join(image_folder, "data_info.pkl"), 'rb') as f:
        data_info = pk.load(f)
        image_names = data_info["image_names"]
        item_names = data_info["item_names"]
        item_lengths = data_info["item_lengths"]
        item_to_image = data_info["item_to_image"]
        #image_to_itemframe = data_info["image_to_itemframe"]

    # Check item_start and item_end
    nitems = len(item_names)
    if not 1<=item_start<=nitems:
        print("Check failed: 1<=item_start<=nitems (1<={0:d}<={1:d})".format(item_start, nitems))
    if item_end == -1:
        item_end = nitems
    elif not 1<=item_end<=nitems:
        print("Check failed: 1<=item_end<=nitems (1<={0:d}<={1:d})".format(item_end, nitems))

    if not exists(dirname(save_path)):
        makedirs(dirname(save_path))


    # ------------------------------------------------------------
    # Run Openpose on each of the items (i.e., video frames and
    # individual images) within image_folder

    results_dict = dict()

    for i in range(item_start-1, item_end):
        item_name = item_names[i]
        print("  Running Openpose on item #{0:d}: {1:s} ...".format(i, item_name))
        item_length = item_lengths[i]

        # Get images_paths from data_info
        image_paths = [None]*item_length
        for k in range(item_length):
            image_paths[k] = join(image_folder, image_names[item_to_image[i] + k])

        if item_length > 1: # video frames
            vis_dir = join(vis_folder, item_name)
        else: # individual images
            vis_dir = vis_folder

        results = main(image_paths, vis_dir, save_path=None)

        # Save estimated joint 2D positions
        results_dict[item_name] = results

        if save_after_each_iteration:
            if exists(save_path):
                with open(save_path, 'rb') as f:
                    data = pk.load(f)
                    data[item_name] = results
            else:
                data = {item_name: results}

            with open(save_path, 'wb') as f:
                pk.dump(data, f)

    if not save_after_each_iteration:
        with open(save_path, 'wb') as f:
            pk.dump(results_dict, f)