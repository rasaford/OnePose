from typing import Tuple

import cv2
import numpy as np

from src.utils.vis_utils import dehomogenize, homogenize


class KeypointTracker:

    def __init__(self, keypoints_3d: np.ndarray):
        self.prev_img = None
        self.active_2d = []
        self.active_3d_idx = []
        self.keypoints_3d = np.asarray(keypoints_3d)
        

    def track(self, img: np.ndarray) -> bool:
        # only track if there sis a valid previous frame
        if self.prev_img is None or len(self.active_2d) <= 2:
            return False

        img_height, img_width = img.shape
        kpts = np.ascontiguousarray(self.active_2d).astype(np.float32)

        # Feature tracker parameters
        lk_params = dict(winSize=(21, 21),
                        maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS , 30 , 0.01))
        tracked_kp, status, err =  cv2.calcOpticalFlowPyrLK(self.prev_img, img, kpts, None, **lk_params)

        valid = status.flatten() == 1 # convert to bool mask
        valid &= (tracked_kp[:, 0] >= 0) & (tracked_kp[:, 1] >= 0) & (tracked_kp[:, 0] < img_height) & (tracked_kp[:, 1] < img_width)
        self.active_2d = list(tracked_kp[valid])
        self.active_3d_idx = [self.active_3d_idx[i] for i, v in enumerate(valid) if v]
        return True


    def transform_2d(self, t_full_to_crop: np.ndarray, pts: np.ndarray) -> np.ndarray:
        # # transform the tracked keypoints to the cropped image
        return dehomogenize((t_full_to_crop @ homogenize(pts).T).T)


    def merge_matches(self, keypoints_2d: np.ndarray, matches: np.ndarray) -> None:

        for i, match_idx in enumerate(matches):
            try:
                kpt_idx = self.active_3d_idx.index(match_idx)
                # this keypoints is already known to the tracker. Since we trust the matching more
                # update it's 2D position to the matched result
                self.active_2d[kpt_idx] = keypoints_2d[i]
            except ValueError:
                # the keypoint is not tarcked already and we start a new tracklet 
                self.active_2d.append(keypoints_2d[i])
                self.active_3d_idx.append(match_idx)


    def  get_active(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the currently active feature tracks

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (kpts_2d, kpts_3d)
        """
        return np.ascontiguousarray(self.active_2d), np.ascontiguousarray(self.keypoints_3d[self.active_3d_idx])

    def get_active_matches(self) -> np.ndarray:
        return np.asarray(self.active_3d_idx)

    def mask_points(self, valid_idx: np.array) -> None:
        """Mask the active tracked points with the given valid indices

        Args:
            valid_idx (np.array): _description_
        """


        # poor mans masking on lists instead of arrays
        self.active_2d = [self.active_2d[idx] for idx in valid_idx]
        self.active_3d_idx = [self.active_3d_idx[idx] for idx in valid_idx]

    def set_prev_image(self, prev_img:np.ndarray):
        self.prev_img = prev_img

    

        