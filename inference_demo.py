import glob
import torch
import hydra
from tqdm import tqdm
import os
import os.path as osp
import cv2
import numpy as np
import natsort

from loguru import logger
from torch.utils.data import DataLoader
from src.utils import data_utils, path_utils, eval_utils, vis_utils
from src.utils.vis_utils import homogenize, dehomogenize
from src.utils.model_io import load_network
from src.local_feature_2D_detector import LocalFeatureObjectDetector


from pytorch_lightning import seed_everything

seed_everything(12345)


def get_default_paths(cfg, data_root, dynamic_dir, sfm_model_dir):
    anno_dir = osp.join(
        sfm_model_dir, f"outputs_{cfg.network.detection}_{cfg.network.matching}", "anno"
    )
    avg_anno_3d_path = osp.join(anno_dir, "anno_3d_average.npz")
    clt_anno_3d_path = osp.join(anno_dir, "anno_3d_collect.npz")
    idxs_path = osp.join(anno_dir, "idxs.npy")
    sfm_ws_dir = osp.join(
        sfm_model_dir,
        f"outputs_{cfg.network.detection}_{cfg.network.matching}",
        "sfm_ws",
        "model",
    )

    img_lists = []
    color_dir = osp.join(dynamic_dir, "color_full")
    img_lists += glob.glob(color_dir + "/*.png", recursive=True)

    img_lists = natsort.natsorted(img_lists)

    # Visualize detector:
    vis_detector_dir = osp.join(dynamic_dir, "detector_vis")
    if osp.exists(vis_detector_dir):
        os.system(f"rm -rf {vis_detector_dir}")
    os.makedirs(vis_detector_dir, exist_ok=True)
    det_box_vis_video_path = osp.join(dynamic_dir, "det_box.mp4")

    # Visualize keypoints:
    keypoint_vis_dir = osp.join(dynamic_dir, "keypoint_vis")
    if osp.exists(keypoint_vis_dir):
        os.system(f"rm -rf {keypoint_vis_dir}")
    os.makedirs(keypoint_vis_dir, exist_ok=True)
    keypoint_vis_video_path = osp.join(dynamic_dir, "keypoint_vis.mp4")

    # Visualize pose:
    vis_box_dir = osp.join(dynamic_dir, "pred_vis")
    if osp.exists(vis_box_dir):
        os.system(f"rm -rf {vis_box_dir}")
    os.makedirs(vis_box_dir, exist_ok=True)

    # save poses
    out_pose_dir = osp.join(dynamic_dir, "poses")
    os.makedirs(out_pose_dir, exist_ok=True)

    demo_video_path = osp.join(dynamic_dir, "demo_video.mp4")

    intrin_full_path = osp.join(dynamic_dir, "intrinsics.txt")
    paths = {
        "data_root": data_root,
        "data_dir": dynamic_dir,
        "sfm_model_dir": sfm_model_dir,
        "sfm_ws_dir": sfm_ws_dir,
        "avg_anno_3d_path": avg_anno_3d_path,
        "clt_anno_3d_path": clt_anno_3d_path,
        "idxs_path": idxs_path,
        "intrin_full_path": intrin_full_path,
        "vis_box_dir": vis_box_dir,
        "vis_detector_dir": vis_detector_dir,
        "det_box_vis_video_path": det_box_vis_video_path,
        "keypoint_vis_dir": keypoint_vis_dir,
        "keypoint_vis_video_path": keypoint_vis_video_path,
        "demo_video_path": demo_video_path,
        "out_pose_dir": out_pose_dir
    }
    return img_lists, paths


def load_model(cfg):
    """Load model"""

    def load_matching_model(model_path):
        """Load onepose model"""
        from src.models.GATsSPG_lightning_model import LitModelGATsSPG

        trained_model = LitModelGATsSPG.load_from_checkpoint(checkpoint_path=model_path)
        print(trained_model.hparams)
        trained_model.cuda()
        trained_model.eval()

        return trained_model

    def load_extractor_model(cfg, model_path):
        """Load extractor model(SuperPoint)"""
        from src.models.extractors.SuperPoint.superpoint import SuperPoint
        from src.sfm.extract_features import confs

        extractor_model = SuperPoint(confs[cfg.network.detection]["conf"])
        extractor_model.cuda()
        extractor_model.eval()
        load_network(extractor_model, model_path, force=True)

        return extractor_model

    matching_model = load_matching_model(cfg.model.onepose_model_path)
    extractor_model = load_extractor_model(cfg, cfg.model.extractor_model_path)
    return matching_model, extractor_model


def load_2D_matching_model(cfg):
    def load_2D_matcher(cfg):
        from src.models.matchers.SuperGlue.superglue import SuperGlue
        from src.sfm.match_features import confs

        match_model = SuperGlue(confs[cfg.network.matching]["conf"])
        match_model.eval()
        load_network(match_model, cfg.model.match_model_path)
        return match_model

    matcher = load_2D_matcher(cfg)
    return matcher


def pack_data(avg_descriptors3d, clt_descriptors, keypoints3d, detection, image_size):
    """Prepare data for OnePose inference"""
    keypoints2d = torch.Tensor(detection["keypoints"])
    descriptors2d = torch.Tensor(detection["descriptors"])

    inp_data = {
        "keypoints2d": keypoints2d[None].cuda(),  # [1, n1, 2]
        "keypoints3d": keypoints3d[None].cuda(),  # [1, n2, 3]
        "descriptors2d_query": descriptors2d[None].cuda(),  # [1, dim, n1]
        "descriptors3d_db": avg_descriptors3d[None].cuda(),  # [1, dim, n2]
        "descriptors2d_db": clt_descriptors[None].cuda(),  # [1, dim, n2*num_leaf]
        "image_size": image_size,
    }

    return inp_data


def inference_core(cfg, data_root, seq_dir, sfm_model_dir):
    """Inference & visualize"""
    from src.datasets.normalized_dataset import NormalizedDataset
    from src.sfm.extract_features import confs
    if cfg.use_tracking:
        from src.tracker.ba_tracker import BATracker
        logger.warning("The tracking module is under development. "
                       "Running OnePose inference without tracking instead.")
        tracker = BATracker(cfg)
        track_interval = 5
    else:
        logger.info("Running OnePose inference without tracking")

    # Load models and prepare data:
    matching_model, extractor_model = load_model(cfg)
    matching_2D_model = load_2D_matching_model(cfg)
    img_lists, paths = get_default_paths(cfg, data_root, seq_dir, sfm_model_dir)

    # sort images
    im_ids = [int(osp.basename(i).replace('.png', '')) for i in img_lists]
    im_ids.sort()
    img_lists = [osp.join(osp.dirname(img_lists[0]), f'{im_id}.png') for im_id in im_ids]

    K, _ = data_utils.get_K(paths["intrin_full_path"])
    box3d_path = path_utils.get_3d_box_path(data_root)
    bbox3d = np.loadtxt(box3d_path)

    local_feature_obj_detector = LocalFeatureObjectDetector(
        extractor_model,
        matching_2D_model,
        sfm_ws_dir=paths["sfm_ws_dir"],
        output_results=True,
        detect_save_dir=paths["vis_detector_dir"],
    )
    dataset = NormalizedDataset(
        img_lists, confs[cfg.network.detection]["preprocessing"]
    )
    loader = DataLoader(dataset, num_workers=1)

    # Prepare 3D features:
    num_leaf = cfg.num_leaf
    avg_data = np.load(paths["avg_anno_3d_path"])
    clt_data = np.load(paths["clt_anno_3d_path"])
    idxs = np.load(paths["idxs_path"])

    keypoints3d = torch.Tensor(clt_data["keypoints3d"]).cuda()
    num_3d = keypoints3d.shape[0]
    # load average 3D features:
    avg_descriptors3d, _ = data_utils.pad_features3d_random(
        avg_data["descriptors3d"], avg_data["scores3d"], num_3d
    )
    # load corresponding 2D features of each 3D point:
    clt_descriptors, _ = data_utils.build_features3d_leaves(
        clt_data["descriptors3d"], clt_data["scores3d"], idxs, num_3d, num_leaf
    )

    pred_poses = {}  # {id:[pred_pose, inliers]}

    # point tracking beween frames
    active_2d_kpts = None
    active_3d_kpts = None
    prev_img = None

    for id, data in enumerate(tqdm(loader, desc="Tracking Frames")):
        with torch.no_grad():
            img_path = data["path"][0]
            inp = data["image"].cuda()
            previous_frame_pose = np.eye(4)

            # Detect object:
            if id == 0:
                # Detect object by 2D local feature matching for the first frame:
                bbox, inp_crop, K_crop, t_full_to_crop = local_feature_obj_detector.detect(inp, img_path, K)
            else:
                # Use 3D bbox and previous frame's pose to yield current frame 2D bbox:
                previous_frame_pose, inliers = pred_poses[id - 1]

                if len(inliers) < 8:
                    # Consider previous pose estimation failed, reuse local feature object detector:
                    bbox, inp_crop, K_crop, t_full_to_crop = local_feature_obj_detector.detect(
                        inp, img_path, K
                    )
                else:
                    (
                        bbox,
                        inp_crop,
                        K_crop,
                        t_full_to_crop
                    ) = local_feature_obj_detector.previous_pose_detect(
                        img_path, K, previous_frame_pose, bbox3d
                    )

            # track all active keypoints
            # if prev_img is not None and active_2d_kpts is not None:
            #     img = (255 * inp.squeeze().cpu().numpy()).astype(np.uint8)
            #     img_height, img_width = img.shape
            #     kpts = np.ascontiguousarray(active_2d_kpts).astype(np.float32)
            #     lk_params = dict(winSize=(21, 21),
            #                     maxLevel=5,
            #                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01))
            #     tracked_kp, status, err =  cv2.calcOpticalFlowPyrLK(prev_img, img, kpts, None, **lk_params)
            #     # transform the tracked keypoints to the cropped image
            #     tracked_kp = dehomogenize((t_full_to_crop @ homogenize(tracked_kp).T).T)
            #     valid = status.flatten() == 1 # convert to bool mask
            #     valid &= (tracked_kp[:, 0] >= 0) & (tracked_kp[:, 1] >= 0) & (tracked_kp[:, 0] < img_height) & (tracked_kp[:, 1] < img_width)
            #     active_2d_kpts = tracked_kp[valid]
            #     active_3d_kpts = active_3d_kpts[valid]
            #     print(f"tracked {len(active_2d_kpts)} 2D keypoints from the previous frame")

            # print(t_full_to_crop.shape)

            # Detect query image(cropped) keypoints and extract descriptors:
            pred_detection = extractor_model(inp_crop)
            pred_detection = {k: v[0].cpu().numpy() for k, v in pred_detection.items()}

            # 2D-3D matching by GATsSPG:
            inp_data = pack_data(
                avg_descriptors3d,
                clt_descriptors,
                keypoints3d,
                pred_detection,
                data["size"],
            )
            pred, _ = matching_model(inp_data)
            matches = pred["matches0"].detach().cpu().numpy()
            valid = matches > -1
            matches = matches[valid]
            kpts2d = pred_detection["keypoints"]
            kpts3d = inp_data["keypoints3d"][0].detach().cpu().numpy()
            confidence = pred["matching_scores0"].detach().cpu().numpy()
            mkpts2d, mkpts3d, mconf = (
                kpts2d[valid],
                kpts3d[matches],
                confidence[valid],
            )

            # if active_2d_kpts is not None:
            #     prev_matches = len(mkpts2d)
            #     mkpts2d = np.concatenate((mkpts2d, active_2d_kpts), axis=0)
            #     matches = np.concatenate((matches, prev_matches + np.arange(active_2d_kpts.shape[0])), axis=0)
            #     mconf = np.concatenate((mconf, np.zeros(shape=(active_2d_kpts.shape[0]))), axis=0)
            # if active_3d_kpts is not None:
            #     mkpts3d = np.concatenate((mkpts3d, active_3d_kpts), axis=0)

            # print(mkpts2d.shape, mkpts3d.shape)


            # Estimate object pose by 2D-3D correspondences:
            pose_pred, pose_pred_homo, inliers = eval_utils.ransac_PnP(
                K_crop, mkpts2d, mkpts3d, scale=1000, initial_pose=previous_frame_pose
            )

            # store matches
            # if len(inliers) > 0:
            #     keep_kpts = np.asarray(inliers).ravel()
            #     t_crop_to_full = np.linalg.inv(t_full_to_crop)
            #     active_2d_kpts = dehomogenize((t_crop_to_full @ homogenize(mkpts2d).T).T)
            #     active_2d_kpts = active_2d_kpts[keep_kpts]
            #     active_3d_kpts = mkpts3d[keep_kpts]
            # else:
            #     active_2d_kpts = None
            #     active_3d_kpts = None

            # Store previous estimated poses:
            pred_poses[id] = [pose_pred, inliers]
            image_crop = np.asarray((inp_crop * 255).squeeze().cpu().numpy(), dtype=np.uint8)

            # visualize the keypoints

            vis_utils.visualize_2d_3d_matches(
                mkpts2d, kpts3d, matches, mconf, pose_pred_homo, K_crop, image_crop, bbox3d,
                img_save_path=osp.join(paths["keypoint_vis_dir"], F"{id}.jpg")
            )

        if cfg.use_tracking and len(inliers) > 8:
            frame_dict = {
                'im_path': image_crop,
                'kpt_pred': pred_detection,
                'pose_pred': pose_pred_homo,
                'pose_gt': pose_pred_homo,
                'K': K_crop,
                'K_crop': K_crop,
                'data': data
            }

            use_update = id % track_interval == 0
            if use_update:
                inliers = np.asarray(inliers)
                mkpts3d_db_inlier = mkpts3d[inliers.flatten()]
                mkpts2d_q_inlier = mkpts2d[inliers.flatten()]

                n_kpt = kpts2d.shape[0]

                valid_query_id = np.where(valid)[0][inliers.flatten()]
                kpts3d_full = np.ones([n_kpt, 3]) * 10086
                kpts3d_full[valid_query_id] = mkpts3d_db_inlier
                kpt3d_ids = matches[valid][inliers.flatten()]

                kf_dict = {
                    'im_path': image_crop,
                    'kpt_pred': pred_detection,
                    'valid_mask': valid,
                    'mkpts2d': mkpts2d_q_inlier,
                    'mkpts3d': mkpts3d_db_inlier,
                    'kpt3d_full': kpts3d_full,
                    'inliers': inliers,
                    'kpt3d_ids': kpt3d_ids,

                    'valid_query_id': valid_query_id,
                    'pose_pred': pose_pred_homo,
                    'pose_gt': pose_pred_homo,
                    'K': K_crop
                }

                need_update = not tracker.update_kf(kf_dict)

            if id == 0:
                tracker.add_kf(kf_dict)
                id += 1
                pose_opt = pose_pred_homo
            else:
                pose_init, pose_opt, ba_log = tracker.track(frame_dict, auto_mode=False)
        else:
            pose_opt = pose_pred_homo

        # Visualize:
        vis_utils.save_demo_image(
            pose_opt,
            K,
            image_path=img_path,
            box3d_path=box3d_path,
            draw_box=len(inliers) > 6,
            save_path=osp.join(paths["vis_box_dir"], f"{id}.jpg"),
        )

        # pose T_co (object to camera)
        np.savetxt(osp.join(paths["out_pose_dir"], f"{id}.txt"), pose_opt)

        # save the previous image for 2D keypoint tracking
        prev_img = (255 * inp.detach().squeeze().cpu().numpy()).astype(np.uint8)


    # Output video to visualize estimated poses:
    vis_utils.make_video(paths["vis_box_dir"], paths["demo_video_path"])
    vis_utils.make_video(paths["keypoint_vis_dir"], paths["keypoint_vis_video_path"])


def inference(cfg):
    data_dirs = cfg.input.data_dirs
    sfm_model_dirs = cfg.input.sfm_model_dirs
    if isinstance(data_dirs, str) and isinstance(sfm_model_dirs, str):
        data_dirs = [data_dirs]
        sfm_model_dirs = [sfm_model_dirs]

    for data_dir, sfm_model_dir in tqdm(
        zip(data_dirs, sfm_model_dirs), total=len(data_dirs)
    ):
        splits = data_dir.split(" ")
        data_root = splits[0]
        for seq_name in splits[1:]:
            seq_dir = osp.join(data_root, seq_name)
            logger.info(f"Eval {seq_dir}")
            inference_core(cfg, data_root, seq_dir, sfm_model_dir)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()
