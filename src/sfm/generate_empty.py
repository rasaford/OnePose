import cv2
import logging
from tqdm import tqdm
import os.path as osp
import numpy as np

from pathlib import Path
from src.utils import path_utils
from src.utils.colmap.read_write_model import Camera, Image, Point3D
from src.utils.colmap.read_write_model import rotmat2qvec
from src.utils.colmap.read_write_model import write_model


def get_pose_from_txt(pose_path):
    """ Read 4x4 transformation matrix from txt """
    pose = np.loadtxt(pose_path)

    tvec = pose[:3, 3].reshape(3, )
    qvec = rotmat2qvec(pose[:3, :3]).reshape(4, )
    return pose, tvec, qvec


def get_intrin_from_txt(intrin_path):
    """ Read 3x3 intrinsic matrix from txt """
    return np.loadtxt(intrin_path)


def import_data(img_lists, pose_paths,  intrin_paths, do_ba=False):
    """ Import intrinsics and camera pose info """

    img_lists = list(map(str, img_lists)) # COLMAP requires str paths

    points3D_out = {}
    images_out = {}
    cameras_out = {}

    def compare(img_name):
        key = img_name.split('/')[-1]
        return int(key.split('.')[0])
    img_lists.sort(key=compare)

    key, img_id, camera_id = 0, 0, 0
    xys_ = np.zeros((0, 2), float)
    point3D_ids_ = np.full(0, -1, int) # will be filled after triangulation

    # import data
    for img_path in tqdm(img_lists, desc="Importing COLMAP"):
        img_path = str(img_path)
        key += 1
        img_id += 1
        camera_id += 1

        img_name = img_path.split('/')[-1]
        img_index = int(img_name.split('.')[0])

        # read pose
        _, tvec, qvec = get_pose_from_txt(pose_paths[img_index])

        # read intrinsic
        K = get_intrin_from_txt(intrin_paths[img_index])
        fx, fy, cx, cy = K[0][0], K[1][1], K[0, 2], K[1, 2]

        image = cv2.imread(img_path)
        h, w, _ = image.shape

        image = Image(
            id=img_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=img_path,
            xys=xys_,
            point3D_ids=point3D_ids_
        )

        camera = Camera(
            id=camera_id,
            model='PINHOLE',
            width=w,
            height=h,
            params=np.array([fx, fy, cx, cy])
        )

        images_out[key] = image
        cameras_out[key] = camera

    return cameras_out, images_out, points3D_out


def generate_model(img_lists, pose_paths, intrin_paths, empty_dir, do_ba=False):
    """ Write intrinsics and camera poses into COLMAP format model"""
    logging.info('Generate empty model...')
    model = import_data(img_lists, pose_paths, intrin_paths, do_ba)

    logging.info(f'Writing the COLMAP model to {empty_dir}')
    Path(empty_dir).mkdir(exist_ok=True, parents=True)
    write_model(*model, path=str(empty_dir), ext='.bin')
    logging.info('Finishing writing model.')
