from .classfication_loss import BCELoss
from .mesh_loss import GANLoss, MeshLoss
from .mse_loss import JointsMSELoss, JointsOHKMMSELoss
from .multi_loss_factory import AELoss, HeatmapLoss, MultiLossFactory
from .regression_loss import L1Loss, MPJPELoss, SmoothL1Loss, WingLoss

__all__ = [
    'JointsMSELoss', 'JointsOHKMMSELoss', 'HeatmapLoss', 'AELoss',
    'MultiLossFactory', 'MeshLoss', 'GANLoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'L1Loss', 'BCELoss'
]
