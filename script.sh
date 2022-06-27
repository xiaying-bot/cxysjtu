python demo/pose_demo_mmdet.py  \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py  \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_cobasco_wholebody_384x288_dark-f5726563_20200918.pth  \
    --img-root /data/sample  \
    --out-img-root /vis_results_whole
