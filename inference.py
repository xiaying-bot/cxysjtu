# Copyright (c) OpenMMLab. All rights reserved.
import os
import time
import warnings
from argparse import ArgumentParser

import cv2

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
# local_runtime = False

# try:
#   from google.colab.patches import cv2_imshow  # for image visualization in colab
# except:
#   local_runtime = True

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# Check Pytorch installation
import torch, torchvision
print('torch version:', torch.__version__, torch.cuda.is_available())
print('torchvision version:', torchvision.__version__)

# Check MMPose installation
import mmpose
print('mmpose version:', mmpose.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print('cuda version:', get_compiling_cuda_version())
print('compiler information:', get_compiler_version())


def main():
    """
    Visualize the demo images/videos.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--is-img', action='store_true', default=False, help='whether to inference on an image.')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument('--img-path', type=str, help='Img path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-root',
        default='',
        help='Root of the output video or img file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)


    if args.is_img:
        print('inference with hrnet on an image:')
        # inference detection
        if args.out_root =='':
            save_out_img = False
        else:
            os.makedirs(args.out_root, exist_ok=True)
            save_out_img = True
        
        mmdet_results = inference_detector(det_model, args.img_path)
        # extract person (COCO_ID=1) bounding boxes from the detection results
        person_results = process_mmdet_results(mmdet_results, cat_id=1)
        print('start inference...')
        pose_results, returned_outputs = inference_top_down_pose_model(pose_model,
                                                               args.img_path,
                                                               person_results,
                                                               bbox_thr=0.3,
                                                               format='xyxy',
                                                               dataset=pose_model.cfg.data.test.type)
        # show pose estimation results
        vis_result = vis_pose_result(pose_model,
                                    args.img_path,
                                    pose_results,
                                    dataset=pose_model.cfg.data.test.type,
                                    show=False)
        # reduce image size
        vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)
        if save_out_img:
            file_name = os.path.join(args.out_root, f'vis_{os.path.basename(args.img_path)}')
            cv2.imwrite(file_name, vis_result)
          
    else:    
        print('inference with hrnet on a video: ')
        cap = cv2.VideoCapture(args.video_path)
        assert cap.isOpened(), f'Faild to load video file {args.video_path}'

        if args.out_root == '':
            save_out_video = False
        else:
            os.makedirs(args.out_root, exist_ok=True)
            save_out_video = True

        if save_out_video:
            fps = cap.get(cv2.CAP_PROP_FPS)
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(
                os.path.join(args.out_root,
                            f'vis_{os.path.basename(args.video_path)}'), fourcc,
                fps, size)

        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None
        print('start inference...')

        while (cap.isOpened()):
            flag, img = cap.read()
            if not flag:
                break
            # test a single image, the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(det_model, img)

            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

            # test a single image, with a list of bboxes.
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                img,
                person_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

            # show the results
            vis_img = vis_pose_result(
                pose_model,
                img,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
                show=False)

            if args.show:
                cv2.imshow('Image', vis_img)

            if save_out_video:
                videoWriter.write(vis_img)

            if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print('inference over.')
        cap.release()
        if save_out_video:
            videoWriter.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
