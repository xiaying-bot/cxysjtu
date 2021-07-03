import numpy as np
import pytest
import torch

from mmpose.apis import (extract_pose_sequence, inference_interhand_3d_model,
                         inference_pose_lifter_model, init_pose_model,
                         vis_3d_pose_result)


def test_pose_lifter_demo():
    # H36M demo
    pose_model = init_pose_model(
        'configs/body/3d_kpt_sview_rgb_img/pose_lift/'
        'h36m/simplebaseline3d_h36m.py',
        None,
        device='cpu')

    pose_det_result = {
        'keypoints': np.zeros((17, 3)),
        'bbox': [50, 50, 50, 50],
        'track_id': 0,
        'image_name': 'tests/data/h36m/S1_Directions_1.54138969_000001.jpg',
    }

    pose_results_2d = [[pose_det_result]]

    dataset = pose_model.cfg.data['test']['type']

    pose_results_2d = extract_pose_sequence(
        pose_results_2d, frame_idx=0, causal=False, seq_len=1, step=1)

    _ = inference_pose_lifter_model(
        pose_model, pose_results_2d, dataset, with_track_id=False)

    pose_lift_results = inference_pose_lifter_model(
        pose_model, pose_results_2d, dataset, with_track_id=True)

    for res in pose_lift_results:
        res['title'] = 'title'
    vis_3d_pose_result(
        pose_model,
        pose_lift_results,
        img=pose_results_2d[0][0]['image_name'],
        dataset=dataset)

    # test special cases
    # Empty 2D results
    _ = inference_pose_lifter_model(
        pose_model, [[]], dataset, with_track_id=False)

    if torch.cuda.is_available():
        _ = inference_pose_lifter_model(
            pose_model.cuda(), pose_results_2d, dataset, with_track_id=False)

    with pytest.raises(NotImplementedError):
        _ = inference_pose_lifter_model(
            pose_model, pose_results_2d, dataset='test')

    # test videopose3d
    pose_model = init_pose_model(
        'configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/'
        'videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py',
        None,
        device='cpu')

    pose_det_result_0 = {
        'keypoints': np.ones((17, 3)),
        'bbox': [50, 50, 100, 100],
        'track_id': 0,
        'image_name': 'tests/data/h36m/S1_Directions_1.54138969_000001.jpg',
    }
    pose_det_result_1 = {
        'keypoints': np.ones((17, 3)),
        'bbox': [50, 50, 100, 100],
        'track_id': 1,
        'image_name': 'tests/data/h36m/S5_SittingDown.54138969_002061.jpg',
    }
    pose_det_result_2 = {
        'keypoints': np.ones((17, 3)),
        'bbox': [50, 50, 100, 100],
        'track_id': 2,
        'image_name': 'tests/data/h36m/S7_Greeting.55011271_000396.jpg',
    }

    pose_results_2d = [[pose_det_result_0], [pose_det_result_1],
                       [pose_det_result_2]]

    dataset = pose_model.cfg.data['test']['type']

    seq_len = pose_model.cfg.test_data_cfg.seq_len
    pose_results_2d_seq = extract_pose_sequence(
        pose_results_2d, 1, causal=False, seq_len=seq_len, step=1)

    pose_lift_results = inference_pose_lifter_model(
        pose_model,
        pose_results_2d_seq,
        dataset,
        with_track_id=True,
        image_size=[1000, 1000],
        norm_pose_2d=True)

    for res in pose_lift_results:
        res['title'] = 'title'
    vis_3d_pose_result(
        pose_model,
        pose_lift_results,
        img=pose_results_2d[0][0]['image_name'],
        dataset=dataset)


def test_interhand3d_demo():
    # H36M demo
    pose_model = init_pose_model(
        'configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/'
        'res50_interhand3d_all_256x256.py',
        None,
        device='cpu')

    image_name = 'tests/data/interhand2.6m/image2017.jpg'
    det_result = {
        'image_name': image_name,
        'bbox': [50, 50, 50, 50],  # bbox format is 'xywh'
        'camera_param': None,
        'keypoints_3d_gt': None
    }
    det_results = [det_result]
    dataset = pose_model.cfg.data['test']['type']

    pose_results = inference_interhand_3d_model(
        pose_model, image_name, det_results, dataset=dataset)

    for res in pose_results:
        res['title'] = 'title'

    vis_3d_pose_result(
        pose_model,
        result=pose_results,
        img=det_results[0]['image_name'],
        dataset=dataset,
    )

    # test special cases
    # Empty det results
    _ = inference_interhand_3d_model(
        pose_model, image_name, [], dataset=dataset)

    if torch.cuda.is_available():
        _ = inference_interhand_3d_model(
            pose_model.cuda(), image_name, det_results, dataset=dataset)

    with pytest.raises(NotImplementedError):
        _ = inference_interhand_3d_model(
            pose_model, image_name, det_results, dataset='test')
