import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl

from sklearn.metrics import confusion_matrix  
import numpy as np

def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)

from stp3.config import get_cfg
from stp3.models.stp3 import STP3
from stp3.losses import SpatialRegressionLoss, SegmentationLoss, HDmapLoss, DepthLoss, ProbabilisticLoss, BEVDetectionLoss, FocalLoss, GaussianFocalLoss
from stp3.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from stp3.utils.geometry import cumulative_warp_features_reverse, cumulative_warp_features
from stp3.utils.instance import predict_instance_segmentation_and_trajectories
from stp3.utils.visualisation import visualise_output

class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # see config.py for details
        self.hparams = hparams
        # pytorch lightning does not support saving YACS CfgNone
        cfg = get_cfg(cfg_dict=self.hparams)
        self.cfg = cfg
        self.n_classes = len(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS)
        self.hdmap_class = cfg.SEMANTIC_SEG.HDMAP.ELEMENTS

        # Bird's-eye view extent in meters
        assert self.cfg.LIFT.X_BOUND[1] > 0 and self.cfg.LIFT.Y_BOUND[1] > 0
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

        # Model
        self.model = STP3(cfg)

        self.losses_fn = nn.ModuleDict()

        # Semantic segmentation
        # self.losses_fn['segmentation'] = SegmentationLoss(
        #     class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS),
        #     use_top_k=self.cfg.SEMANTIC_SEG.VEHICLE.USE_TOP_K,
        #     top_k_ratio=self.cfg.SEMANTIC_SEG.VEHICLE.TOP_K_RATIO,
        #     future_discount=self.cfg.FUTURE_DISCOUNT,
        # )
        self.losses_fn['segmentation'] = FocalLoss(class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS))
        self.model.segmentation_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.metric_vehicle_val = IntersectionOverUnion(self.n_classes)

        if self.cfg.SEMANTIC_SEG.SUBGOAL.ENABLED:
            self.model.subgoal_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            # self.model.subgoal_offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.losses_fn['subgoal'] = GaussianFocalLoss()
            self.metric_subgoal_val = IntersectionOverUnion(self.n_classes)

        # Pedestrian segmentation
        if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            self.model.pedestrian_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            if self.cfg.USE_DETECTION:
                self.model.pedestrian_offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
                self.losses_fn['pedestrian'] = BEVDetectionLoss()
            else:
                self.losses_fn['pedestrian'] = SegmentationLoss(
                    class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.PEDESTRIAN.WEIGHTS),
                    use_top_k=self.cfg.SEMANTIC_SEG.PEDESTRIAN.USE_TOP_K,
                    top_k_ratio=self.cfg.SEMANTIC_SEG.PEDESTRIAN.TOP_K_RATIO,
                    future_discount=self.cfg.FUTURE_DISCOUNT,
                )
            self.metric_pedestrian_val = IntersectionOverUnion(self.n_classes)

        # HD map
        if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            # self.losses_fn['hdmap'] = HDmapLoss(
            #     class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.HDMAP.WEIGHTS),
            #     training_weights=self.cfg.SEMANTIC_SEG.HDMAP.TRAIN_WEIGHT,
            #     use_top_k=self.cfg.SEMANTIC_SEG.HDMAP.USE_TOP_K,
            #     top_k_ratio=self.cfg.SEMANTIC_SEG.HDMAP.TOP_K_RATIO,
            # )
            # self.metric_hdmap_val = []
            # for i in range(len(self.hdmap_class)):
            #     self.metric_hdmap_val.append(IntersectionOverUnion(2, absent_score=1))
            # self.model.hdmap_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            # self.metric_hdmap_val = nn.ModuleList(self.metric_hdmap_val)
            # Lane segmentation
            if self.cfg.SEMANTIC_SEG.LANE.ENABLED:
                self.losses_fn['lane'] = SegmentationLoss(
                    class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.LANE.WEIGHTS),
                    use_top_k=self.cfg.SEMANTIC_SEG.LANE.USE_TOP_K,
                    top_k_ratio=self.cfg.SEMANTIC_SEG.LANE.TOP_K_RATIO,
                    future_discount=1.0,
                )
                self.model.lane_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
                self.metric_lane_val = IntersectionOverUnion(2)
            
            # Area segmentation
            if self.cfg.SEMANTIC_SEG.AREA.ENABLED:
                # self.losses_fn['area'] = SegmentationLoss(
                #     class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.AREA.WEIGHTS),
                #     use_top_k=self.cfg.SEMANTIC_SEG.AREA.USE_TOP_K,
                #     top_k_ratio=self.cfg.SEMANTIC_SEG.AREA.TOP_K_RATIO,
                #     future_discount=1.0,
                # )
                self.losses_fn['area'] = FocalLoss(class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.AREA.WEIGHTS))
                self.model.area_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
                self.metric_area_val = IntersectionOverUnion(2)

        if self.cfg.SEMANTIC_SEG.OBSTACLE.ENABLED:
            # self.losses_fn['obstacle'] = SegmentationLoss(
            #     class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.OBSTACLE.WEIGHTS),
            #     use_top_k=self.cfg.SEMANTIC_SEG.OBSTACLE.USE_TOP_K,
            #     top_k_ratio=self.cfg.SEMANTIC_SEG.OBSTACLE.TOP_K_RATIO,
            #     future_discount=self.cfg.FUTURE_DISCOUNT,
            # )
            self.losses_fn['obstacle'] = FocalLoss(class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.OBSTACLE.WEIGHTS))
            self.model.obstacle_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.metric_obstacle_val = IntersectionOverUnion(self.n_classes)

        if self.cfg.SEMANTIC_SEG.TRAFFIC.ENABLED:
            self.losses_fn['traffic'] = FocalLoss()
            self.model.traffic_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        if self.cfg.SEMANTIC_SEG.STOP.ENABLED:
            self.model.stop_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            # if self.cfg.USE_DETECTION:
            #     self.model.stop_offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            #     self.losses_fn['stop'] = BEVDetectionLoss()
            # else:
            #     self.losses_fn['stop'] = SegmentationLoss(
            #     class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.STOP.WEIGHTS),
            #     use_top_k=self.cfg.SEMANTIC_SEG.STOP.USE_TOP_K,
            #     top_k_ratio=self.cfg.SEMANTIC_SEG.STOP.TOP_K_RATIO,
            #     future_discount=1.0,
            #     )
            self.losses_fn['stop'] = FocalLoss(class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.STOP.WEIGHTS))
            self.metric_stop_val = IntersectionOverUnion(2)

        # Depth
        if self.cfg.LIFT.GT_DEPTH:
            self.losses_fn['depths'] = DepthLoss()
            self.model.depths_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Instance segmentation
        if self.cfg.INSTANCE_SEG.ENABLED:
            self.losses_fn['instance_center'] = SpatialRegressionLoss(
                norm=2, future_discount=self.cfg.FUTURE_DISCOUNT
            )
            self.losses_fn['instance_offset'] = SpatialRegressionLoss(
                norm=1, future_discount=self.cfg.FUTURE_DISCOUNT, ignore_index=self.cfg.DATASET.IGNORE_INDEX
            )
            self.model.centerness_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.model.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.metric_panoptic_val = PanopticMetric(n_classes=self.n_classes)

        # Instance flow
        if self.cfg.INSTANCE_FLOW.ENABLED:
            self.losses_fn['instance_flow'] = SpatialRegressionLoss(
                norm=1, future_discount=self.cfg.FUTURE_DISCOUNT, ignore_index=self.cfg.DATASET.IGNORE_INDEX
            )
            self.model.flow_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Planning
        if self.cfg.PLANNING.ENABLED:
            self.metric_planning_val = PlanningMetric(self.cfg, self.cfg.N_FUTURE_FRAMES)
            self.model.planning_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        if self.cfg.PROBABILISTIC.ENABLED and self.cfg.PROBABILISTIC.GT_FUTURE:
            self.losses_fn['probabilistic'] = ProbabilisticLoss(self.cfg.PROBABILISTIC.METHOD)
            self.model.prob_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.training_step_count = 0

    def shared_step(self, batch, is_train):
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']
        affine_mats = batch['affine_mats']
        command = batch['command']
        trajs = batch['sample_trajectory']
        target_points = batch['target_point']
        lidar = batch['lidar']

        # Warp labels
        labels, future_inputs = self.prepare_future_labels(batch)
        future_inputs = future_inputs if is_train and self.cfg.PROBABILISTIC.GT_FUTURE else None
        target_point = command = None
        if self.cfg.SEMANTIC_SEG.SUBGOAL.ENABLED:
            target_point, command = labels['target_point'], labels['command']

        # Forward pass
        output = self.model(
            image, intrinsics, extrinsics, future_egomotion, affine_mats, 
            future_inputs, lidar, is_train, labels['depths'], target_point, command
        )

        #####
        # Loss computation
        #####
        loss = {'total': 0}
        n_present = self.model.receptive_field
        start = 0
        if self.cfg.N_FUTURE_FRAMES > 0 and self.cfg.PRETRAINED.PATH != "" \
            and self.cfg.PRETRAINED.FREEZE:
            start = n_present - 1
        if is_train:
            # segmentation
            self.model.segmentation_weight.data.clamp_(-6, None)
            segmentation_factor = 1 / (2 * torch.exp(self.model.segmentation_weight))
            seg_predictions = output['segmentation'][:, start:]
            seg_labels = labels['segmentation'][:, start:]
            seg_masks = labels['segmentation_occ'][:, start:]
            # if self.cfg.TRUNCATE != 0:
            #     seg_predictions = seg_predictions[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
            #     seg_labels = seg_labels[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
            #     seg_masks = seg_masks[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
            loss['segmentation'] = self.losses_fn['segmentation'](seg_predictions, seg_labels, seg_masks)
            loss['segmentation_uncertainty'] = 0.5 * self.model.segmentation_weight
            loss['total'] = loss['total'] + \
                segmentation_factor * loss['segmentation'] + \
                    loss['segmentation_uncertainty']
            
            if self.cfg.SEMANTIC_SEG.SUBGOAL.ENABLED:
                self.model.subgoal_weight.data.clamp_(-6, None)
                subgoal_factor = 1 / (2 * torch.exp(self.model.subgoal_weight))
                # subgoal_center_factor = 1 / (2 * torch.exp(self.model.subgoal_weight))
                # self.model.subgoal_offset_weight.data.clamp_(-6, None)
                # subgoal_offset_factor = 1 / (2 * torch.exp(self.model.subgoal_offset_weight))

                sub_predictions = output['subgoal_heat']
                sub_labels = labels['subgoal_heat']
                loss['subgoal'] = self.losses_fn['subgoal'](sub_predictions, sub_labels)
                loss['subgoal_uncertainty'] = 0.5 * self.model.subgoal_weight
                loss['total'] = loss['total'] + \
                    subgoal_factor * loss['subgoal'] + loss['subgoal_uncertainty']
                # loss['subgoal_center'], loss['subgoal_offset'] = \
                #     self.losses_fn['subgoal'](**sub_predictions, **sub_labels)
                # loss['subgoal_center_uncertainty'] = 0.5 * self.model.subgoal_weight
                # loss['subgoal_offset_uncertainty'] = 0.5 * self.model.subgoal_offset_weight
                # loss['total'] = loss['total'] + \
                #     subgoal_center_factor * loss['subgoal_center'] + \
                #     subgoal_offset_factor * loss['subgoal_offset'] + \
                #         loss['subgoal_center_uncertainty'] + loss['subgoal_offset_uncertainty']

            # Pedestrian
            if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
                if self.cfg.USE_DETECTION:
                    self.model.pedestrian_weight.data.clamp_(-6, None)
                    pedestrian_center_factor = 1 / (2 * torch.exp(self.model.pedestrian_weight))
                    self.model.pedestrian_offset_weight.data.clamp_(-6, None)
                    pedestrian_offset_factor = 1 / (2 * torch.exp(self.model.pedestrian_offset_weight))

                    ped_predictions = output['pedestrian_detection']
                    ped_labels = labels['pedestrian_detection']
                    loss['pedestrian_center'], loss['pedestrian_offset'] = \
                        self.losses_fn['pedestrian'](**ped_predictions, **ped_labels)
                    loss['pedestrian_center_uncertainty'] = 0.5 * self.model.pedestrian_weight
                    loss['pedestrian_offset_uncertainty'] = 0.5 * self.model.pedestrian_offset_weight
                    loss['total'] = loss['total'] + \
                        pedestrian_center_factor * loss['pedestrian_center'] + \
                        pedestrian_offset_factor * loss['pedestrian_offset'] + \
                            loss['pedestrian_center_uncertainty'] + loss['pedestrian_offset_uncertainty']
                else:
                    self.model.pedestrian_weight.data.clamp_(-6, None)
                    pedestrian_factor = 1 / (2 * torch.exp(self.model.pedestrian_weight))
                    ped_predictions = output['pedestrian'][:, start:]
                    ped_labels = labels['pedestrian'][:, start:]
                    ped_masks = labels['pedestrian_occ'][:, start:]
                    # if self.cfg.TRUNCATE != 0:
                    #     ped_predictions = ped_predictions[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                    #     ped_labels = ped_labels[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                    #     ped_masks = ped_masks[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                    loss['pedestrian'] = self.losses_fn['pedestrian'](ped_predictions, ped_labels, ped_masks)
                    loss['pedestrian_uncertainty'] = 0.5 * self.model.pedestrian_weight
                    loss['total'] = loss['total'] + \
                        pedestrian_factor * loss['pedestrian'] + \
                            loss['pedestrian_uncertainty']
                    
            # hdmap loss
            if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
                # hdmap_factor = 1 / (2 * torch.exp(self.model.hdmap_weight))
                # loss['hdmap'] = self.losses_fn['hdmap'](output['hdmap'], labels['hdmap'])
                # loss['hdmap_uncertainty'] = 0.5 * self.model.hdmap_weight
                # loss['total'] = loss['total'] + \
                #     hdmap_factor * loss['hdmap'] + \
                #         loss['hdmap_uncertainty']
            
                if self.cfg.SEMANTIC_SEG.LANE.ENABLED and not self.cfg.PRETRAINED.FREEZE:
                    self.model.lane_weight.data.clamp_(-6, None)
                    lane_factor = 1 / (2 * torch.exp(self.model.lane_weight))
                    lane_predictions = output['lane'][:, start:n_present]
                    lane_labels = labels['lane'][:, start:n_present]
                    # if self.cfg.TRUNCATE != 0:
                    #     lane_predictions = lane_predictions[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                    #     lane_labels = lane_labels[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                    loss['lane'] = self.losses_fn['lane'](lane_predictions, lane_labels)
                    loss['lane_uncertainty'] = 0.5 * self.model.lane_weight
                    loss['total'] = loss['total'] + \
                        lane_factor * loss['lane'] + \
                            loss['lane_uncertainty']
                
                if self.cfg.SEMANTIC_SEG.AREA.ENABLED and not self.cfg.PRETRAINED.FREEZE:
                    self.model.area_weight.data.clamp_(-6, None)
                    area_factor = 1 / (2 * torch.exp(self.model.area_weight))
                    area_predictions = output['area'][:, start:n_present]
                    area_labels = labels['area'][:, start:n_present]
                    # if self.cfg.TRUNCATE != 0:
                    #     area_predictions = area_predictions[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                    #     area_labels = area_labels[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                    loss['area'] = self.losses_fn['area'](area_predictions, area_labels)
                    loss['area_uncertainty'] = 0.5 * self.model.area_weight
                    loss['total'] = loss['total'] + \
                        area_factor * loss['area'] + \
                            loss['area_uncertainty']

            if self.cfg.SEMANTIC_SEG.OBSTACLE.ENABLED:
                self.model.obstacle_weight.data.clamp_(-6, None)
                obstacle_factor = 1 / (2 * torch.exp(self.model.obstacle_weight))
                obstacle_predictions = output['obstacle'][:, start:n_present]
                obstacle_labels = labels['obstacle'][:, start:n_present]
                # if self.cfg.TRUNCATE != 0:
                #     area_predictions = area_predictions[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                #     area_labels = area_labels[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                loss['obstacle'] = self.losses_fn['obstacle'](obstacle_predictions, obstacle_labels)
                loss['obstacle_uncertainty'] = 0.5 * self.model.obstacle_weight
                loss['total'] = loss['total'] + \
                    area_factor * loss['obstacle'] + \
                        loss['obstacle_uncertainty']
            
            if self.cfg.SEMANTIC_SEG.TRAFFIC.ENABLED:
                self.model.traffic_weight.data.clamp_(-6, None)
                traffic_factor = 1 / (2 * torch.exp(self.model.traffic_weight))
                traffic_predictions = output['traffic'][:, start:n_present]
                traffic_labels = labels['traffic'][:, start:n_present]
                # if self.cfg.TRUNCATE != 0:
                #     area_predictions = area_predictions[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                #     area_labels = area_labels[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                loss['traffic'] = self.losses_fn['traffic'](traffic_predictions.view(-1, 2), traffic_labels.view(-1).long())
                # print(traffic_predictions.view(-1, 2)[:, :1])
                # print(traffic_labels.view(-1, 1))
                
                # loss['traffic'] = torch.nn.functional.cross_entropy(
                #     traffic_predictions.view(-1, 2),
                #     traffic_labels.view(-1).long()
                # )
                # print(traffic_predictions.view(-1, 2))
                # print(traffic_predictions.view(-1, 2).argmax(dim=1))
                # print(traffic_labels.view(-1, 1))
                # print(loss['traffic'])
                loss['traffic_uncertainty'] = 0.5 * self.model.traffic_weight
                loss['total'] = loss['total'] + \
                    area_factor * loss['traffic'] + \
                        loss['traffic_uncertainty']

            if self.cfg.SEMANTIC_SEG.STOP.ENABLED and not self.cfg.PRETRAINED.FREEZE:
                # if self.cfg.USE_DETECTION:
                #     self.model.stop_weight.data.clamp_(-6, None)
                #     stop_center_factor = 1 / (2 * torch.exp(self.model.stop_weight))
                #     self.model.stop_offset_weight.data.clamp_(-6, None)
                #     stop_offset_factor = 1 / (2 * torch.exp(self.model.stop_offset_weight))

                #     stop_predictions = output['stop_detection']
                #     stop_labels = labels['stop_detection']
                #     loss['stop_center'], loss['stop_offset'] = \
                #         self.losses_fn['stop'](**stop_predictions, **stop_labels)
                #     loss['stop_center_uncertainty'] = 0.5 * self.model.stop_weight
                #     loss['stop_offset_uncertainty'] = 0.5 * self.model.stop_offset_weight
                #     loss['total'] = loss['total'] + \
                #         stop_center_factor * loss['stop_center'] + \
                #         stop_offset_factor * loss['stop_offset'] + \
                #             loss['stop_center_uncertainty'] + loss['stop_offset_uncertainty']
                # else:
                self.model.stop_weight.data.clamp_(-6, None)
                stop_factor = 1 / (2 * torch.exp(self.model.stop_weight))
                stop_predictions = output['stop'][:, start:n_present]
                stop_labels = labels['stop'][:, start:n_present]
                # stop_masks = labels['stop_occ'][:, start:n_present]
                # if self.cfg.TRUNCATE != 0:
                #     stop_predictions = stop_predictions[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                #     stop_labels = stop_labels[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                #     stop_masks = stop_masks[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                loss['stop'] = self.losses_fn['stop'](stop_predictions, stop_labels)
                loss['stop_uncertainty'] = 0.5 * self.model.stop_weight
                loss['total'] = loss['total'] + \
                    stop_factor * loss['stop'] + \
                        loss['stop_uncertainty']

            # if self.cfg.INSTANCE_SEG.ENABLED:
            #     # instance center
            #     centerness_factor = 1 / (2 * torch.exp(self.model.centerness_weight))
            #     loss['instance_center'] = centerness_factor * self.losses_fn['instance_center'](
            #         output['instance_center'], labels['centerness'], self.model.receptive_field
            #     )
            #     loss['centerness_uncertainty'] = 0.5 * self.model.centerness_weight

            #     # instance offset
            #     offset_factor = 1 / (2 * torch.exp(self.model.offset_weight))
            #     loss['instance_offset'] = offset_factor * self.losses_fn['instance_offset'](
            #         output['instance_offset'], labels['offset'], self.model.receptive_field
            #     )
            #     loss['offset_uncertainty'] = 0.5 * self.model.offset_weight
            if self.cfg.PROBABILISTIC.ENABLED and self.cfg.PROBABILISTIC.GT_FUTURE:
                self.model.prob_weight.data.clamp_(-6, None)
                prob_factor = 1 / (2 * torch.exp(self.model.prob_weight))
                loss['prob'] = self.losses_fn['probabilistic'](output['output_distribution'])
                loss['prob_uncertainty'] = 0.5 * self.model.prob_weight 
                loss['total'] = loss['total'] + \
                    prob_factor * loss['prob'] + \
                        loss['prob_uncertainty']

            # depth loss
            if self.cfg.LIFT.GT_DEPTH:
                self.model.depths_weight.data.clamp_(-6, None)
                depths_factor = 1 / (2 * torch.exp(self.model.depths_weight))
                loss['depths'] = self.losses_fn['depths'](output['depth_prediction'], labels['depths'])
                loss['depths_uncertainty'] = 0.5 * self.model.depths_weight
                loss['total'] = loss['total'] + \
                    depths_factor * loss['depths'] + \
                        loss['depths_uncertainty']
            # # instance flow
            # if self.cfg.INSTANCE_FLOW.ENABLED:
            #     flow_factor = 1 / (2 * torch.exp(self.model.flow_weight))
            #     loss['instance_flow'] = flow_factor * self.losses_fn['instance_flow'](
            #         output['instance_flow'], labels['flow'], self.model.receptive_field
            #     )
            #     loss['flow_uncertainty'] = 0.5 * self.model.flow_weight

            # Planning
            if self.cfg.PLANNING.ENABLED:
                receptive_field = self.model.receptive_field
                self.model.planning_weight.data.clamp_(-6, None)
                planning_factor = 1 / (2 * torch.exp(self.model.planning_weight))
                occupancy = torch.logical_or(labels['segmentation'][:, receptive_field:].squeeze(2),
                                             labels['pedestrian'][:, receptive_field:].squeeze(2))
                pl_loss, final_traj = self.model.planning(
                    cam_front=output['cam_front'].detach(),
                    trajs=trajs[:, :, 1:],
                    gt_trajs=labels['gt_trajectory'][:, 1:],
                    cost_volume=output['costvolume'][:, receptive_field:],
                    semantic_pred=occupancy,
                    hd_map=labels['hdmap'],
                    commands=command,
                    target_points=target_points
                )
                loss['planning'] = pl_loss
                loss['planning_uncertainty'] = 0.5 * self.model.planning_weight
                loss['total'] = loss['total'] + \
                    planning_factor * loss['planning'] + \
                        loss['planning_uncertainty']
                output = {**output, 'selected_traj': torch.cat(
                    [torch.zeros((B, 1, 3), device=final_traj.device), final_traj], dim=1)}
            else:
                output = {**output, 'selected_traj': labels['gt_trajectory']}

        # Metrics
        else:
            # semantic segmentation metric
            seg_predictions = output['segmentation'].detach()
            seg_predictions = torch.argmax(seg_predictions, dim=2, keepdim=True)
            seg_predictions = seg_predictions[:, start:]*labels['segmentation_occ'][:, start:]
            seg_labels = labels['segmentation'][:, start:]*labels['segmentation_occ'][:, start:]
            # if self.cfg.TRUNCATE != 0:
            #     seg_predictions = seg_predictions[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
            #     seg_labels = seg_labels[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
            self.metric_vehicle_val(seg_predictions, seg_labels)

            if self.cfg.SEMANTIC_SEG.SUBGOAL.ENABLED:
                sub_predictions = output['subgoal'].detach()
                sub_labels = labels['subgoal']
                self.metric_subgoal_val(sub_predictions, sub_labels)

            # pedestrian segmentation metric
            if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
                if self.cfg.USE_DETECTION:
                    ped_predictions = output['pedestrian'].detach()
                    ped_predictions = ped_predictions[:, start:]*labels['pedestrian_occ'][:, start:]
                else:
                    ped_predictions = output['pedestrian'].detach()
                    ped_predictions = torch.argmax(ped_predictions, dim=2, keepdim=True)
                    ped_predictions = ped_predictions[:, start:]*labels['pedestrian_occ'][:, start:]
                ped_labels = labels['pedestrian'][:, start:]*labels['pedestrian_occ'][:, start:]
                # if self.cfg.TRUNCATE != 0:
                #     ped_predictions = ped_predictions[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                #     ped_labels = ped_labels[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                self.metric_pedestrian_val(ped_predictions, ped_labels)
            else:
                pedestrian_prediction = torch.zeros_like(seg_prediction)

            # hdmap metric
            if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
                # for i in range(len(self.hdmap_class)):
                #     hdmap_prediction = output['hdmap'][:, 2 * i:2 * (i + 1)].detach()
                #     hdmap_prediction = torch.argmax(hdmap_prediction, dim=1, keepdim=True)
                #     self.metric_hdmap_val[i](hdmap_prediction, labels['hdmap'][:, i:i + 1])
                if self.cfg.SEMANTIC_SEG.LANE.ENABLED:
                    lane_predictions = output['lane'].detach()
                    lane_predictions = torch.argmax(lane_predictions, dim=2, keepdim=True)
                    lane_predictions = lane_predictions[:, start:n_present]
                    lane_labels = labels['lane'][:, start:n_present]
                    # if self.cfg.TRUNCATE != 0:
                    #     lane_predictions = lane_predictions[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                    #     lane_labels = lane_labels[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                    self.metric_lane_val(lane_predictions, lane_labels)

                if self.cfg.SEMANTIC_SEG.AREA.ENABLED:
                    area_predictions = output['area'].detach()
                    area_predictions = torch.argmax(area_predictions, dim=2, keepdim=True)
                    area_predictions = area_predictions[:, start:n_present]
                    area_labels = labels['area'][:, start:n_present]
                    # if self.cfg.TRUNCATE != 0:
                    #     area_predictions = area_predictions[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                    #     area_labels = area_labels[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                    self.metric_area_val(area_predictions, area_labels)

            if self.cfg.SEMANTIC_SEG.OBSTACLE.ENABLED:
                obstacle_predictions = output['obstacle'].detach()
                obstacle_predictions = torch.argmax(obstacle_predictions, dim=2, keepdim=True)
                obstacle_predictions = obstacle_predictions[:, start:n_present]
                obstacle_labels = labels['obstacle'][:, start:n_present]
                # if self.cfg.TRUNCATE != 0:
                #     area_predictions = area_predictions[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                #     area_labels = area_labels[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                self.metric_obstacle_val(obstacle_predictions, obstacle_labels)

            if self.cfg.SEMANTIC_SEG.STOP.ENABLED:
                # if self.cfg.USE_DETECTION:
                #     stop_predictions = output['stop'].detach()
                #     stop_predictions = stop_predictions[:, start:n_present]
                # else:
                stop_predictions = output['stop'].detach()
                stop_predictions = torch.argmax(stop_predictions, dim=2, keepdim=True)
                stop_predictions = stop_predictions[:, start:n_present]
                stop_labels = labels['stop'][:, start:n_present]
                # if self.cfg.TRUNCATE != 0:
                #     stop_predictions = stop_predictions[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                #     stop_labels = stop_labels[..., self.cfg.TRUNCATE-1:-self.cfg.TRUNCATE]
                self.metric_stop_val(stop_predictions, stop_labels)

            # instance segmentation metric
            if self.cfg.INSTANCE_SEG.ENABLED:
                pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                    output, compute_matched_centers=False
                )
                self.metric_panoptic_val(pred_consistent_instance_seg[:, n_present - 1:],
                                         labels['instance'][:, n_present - 1:])

            # planning metric
            if self.cfg.PLANNING.ENABLED:
                occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
                _, final_traj = self.model.planning(
                    cam_front=output['cam_front'].detach(),
                    trajs=trajs[:, :, 1:],
                    gt_trajs=labels['gt_trajectory'][:, 1:],
                    cost_volume=output['costvolume'][:, n_present:].detach(),
                    semantic_pred=occupancy[:, n_present:].squeeze(2),
                    hd_map=output['hdmap'].detach(),
                    commands=command,
                    target_points=target_points
                )
                occupancy = torch.logical_or(labels['segmentation'][:, n_present:].squeeze(2),
                                             labels['pedestrian'][:, n_present:].squeeze(2))
                self.metric_planning_val(final_traj, labels['gt_trajectory'][:, 1:], occupancy)
                output = {**output,
                          'selected_traj': torch.cat([torch.zeros((B, 1, 3), device=final_traj.device), final_traj],
                                                     dim=1)}
            else:
                output = {**output, 'selected_traj': labels['gt_trajectory']}

        return output, labels, loss

    def prepare_future_labels(self, batch):
        labels = {}
        future_inputs = []

        segmentation_labels = batch['segmentation']
        invisible_mask = 1 - batch['visible_mask']
        hdmap_labels = batch['hdmap']
        stop_labels = batch['stop_area']
        # labels['stop_detection'] = \
        #     {'center_heatmap_target': batch['stop_area_heatmap'].contiguous(),
        #         'offset_target': batch['stop_area_offset'].contiguous(),
        #         'pixel_mask': batch['stop_area_pixel_mask'].contiguous()}
        labels['pedestrian_detection'] = \
            {'center_heatmap_target': batch['pedestrian_heatmap'].contiguous(),
                'offset_target': batch['pedestrian_offset'].contiguous(),
                'pixel_mask': batch['pedestrian_pixel_mask'].contiguous()}
        # labels['subgoal_detection'] = \
        #     {'center_heatmap_target': batch['subgoal_heatmap'][:, :5].unsqueeze(2).contiguous(),
        #         'offset_target': batch['subgoal_offset'][:, :5].contiguous(),
        #         'pixel_mask': batch['subgoal_pixel_mask'][:, :5].unsqueeze(2).contiguous()}
        # for k in labels['subgoal_detection']:
        #     print(k, labels['subgoal_detection'][k].shape)
        labels['subgoal_heat'] = batch['subgoal_heatmap'][:, :5].float().contiguous()
        labels['subgoal'] = batch['subgoal_dec'][:, :5].long().contiguous()

        labels['visible_mask'] = batch['visible_mask']
        labels['traffic'] = batch['is_stop_moment'].float()

        # to be removed
        labels['depths'] = None
        # if True:
        if self.model.visualize:
            # to be removed
            print(batch['path'])
            colors = {
            "black": (0, 0, 0),
            "silver": (192, 192, 192),
            "gray": (128, 128, 128),
            "white": (255, 255, 255),
            "maroon": (128, 0, 0),
            "red": (255, 0, 0),
            "purple": (128, 0, 128),
            "fuchsia": (255, 0, 255),
            "green": (0, 128, 0),
            "lime": (0, 255, 0),
            "olive": (128, 128, 0),
            "yellow": (255, 255, 0),
            "navy": (0, 0, 128),
            "blue": (0, 0, 255),
            "teal": (0, 128, 128),
            "aqua": (0, 255, 255)
            }
            sem = torch.zeros(256, 256, 3)
            nonzero_indices = lambda arr: arr != 0
            area = hdmap_labels[0, self.model.receptive_field - 1, 1]
            vehicle = segmentation_labels[0, self.model.receptive_field - 1, 0]
            target = batch['target_point_dec'][0, 0]
            subgoal = batch['subgoal_dec'][0, 0, 0]
            subgoal2 = batch['subgoal_dec'][0, 0, -1]
            print(area.shape, vehicle.shape, target.shape, subgoal.shape)
            print(batch['command_feat'], batch['command'], batch['target_point'])
            for k in batch:
                if 'subgoal' in k:
                    print(k, batch[k].shape)
            sem[nonzero_indices(area)] = torch.Tensor(list(colors['gray']))
            sem[nonzero_indices(vehicle)] = torch.Tensor(list(colors['teal']))
            sem[nonzero_indices(target)] = torch.Tensor(list(colors['maroon']))
            sem[nonzero_indices(subgoal)] = torch.Tensor(list(colors['navy']))
            sem[nonzero_indices(subgoal2)] = torch.Tensor(list(colors['olive']))

            # lst = ["navy", "teal", "red", "purple", "olive", 'green', 'maroon']
            # for objs in segmentation_labels:
            #     lst_ = lst[:len(objs)]
            #     for t, c in zip(range(len(objs)), lst_):
            #         # sem[nonzero_indices(objs[t, 0])] = torch.Tensor(list(colors[c]))
            #         sem[nonzero_indices(batch['stop_area'][0, 0, 0])] = torch.Tensor(list(colors['green']))
            #         break
            torchvision.utils.save_image(sem.permute(2, 0, 1), f'tmp_test/warp.png')
            # sem = torch.zeros(256, 256, 3)
            # nonzero_indices = lambda arr: arr != 0
            # # sem[nonzero_indices(hdmap_labels[0, -1, 1])] = torch.Tensor(list(colors['gray']))
            # # sem[nonzero_indices(hdmap_labels[0, -1, 0])] = torch.Tensor(list(colors['green']))
            # # sem[nonzero_indices(hdmap_labels[0, 0, 0])] = torch.Tensor(list(colors['maroon']))
            # for objs in segmentation_labels:
            #     lst_ = lst[:len(objs)]
            #     for t, c in zip(range(len(objs)), lst_):
            #         # sem[nonzero_indices(objs[t, 0])] = torch.Tensor(list(colors[c]))
            #         sem[nonzero_indices(batch['stop_area_dec'][0, 0, 0])] = torch.Tensor(list(colors['green']))
            #         break
            # torchvision.utils.save_image(sem.permute(2, 0, 1), f'tmp_test/warp2.png')
            # torchvision.utils.save_image(batch['pedestrian_heatmap'][0,0,0], f'tmp_test/heat.png')
            # for objs in segmentation_occ:
            #     for t in range(len(objs)):
            #         sem[nonzero_indices(objs[t, 0])] = torch.Tensor(list(colors['black']))
            # torchvision.utils.save_image(sem.permute(2, 0, 1), f'tmp_test/warp_masked.png')
            
            depths = batch['depths']
            depth_labels = depths[:, :self.model.receptive_field]
            depth_labels = torch.clamp(depth_labels, self.cfg.LIFT.D_BOUND[0], self.cfg.LIFT.D_BOUND[1] - 1) - \
                            self.cfg.LIFT.D_BOUND[0]
            depth_labels = depth_labels.long().contiguous()
            labels['depths'] = depth_labels

        future_egomotion = batch['future_egomotion']
        gt_trajectory = batch['gt_trajectory']

        # present frame hd map gt
        labels['hdmap'] = hdmap_labels[:, self.model.receptive_field - 1].long().contiguous()
        labels['lane'] = torch.cat([hdmap_labels[:, self.model.receptive_field - 1, :1].unsqueeze(1)] * \
            hdmap_labels.shape[1], dim=1).long().contiguous()
        labels['area'] = torch.cat([hdmap_labels[:, self.model.receptive_field - 1, 1:].unsqueeze(1)] * \
            hdmap_labels.shape[1], dim=1).long().contiguous()
        # if self.cfg.USE_DETECTION:
        #     labels['stop'] = batch['stop_area_dec'].long().contiguous()
        # else:
        labels['stop'] = stop_labels.long().contiguous()
        labels['stop_occ'] = (1 - stop_labels * \
                invisible_mask[:, :self.model.receptive_field]).long().contiguous().float()

        # gt trajectory
        labels['gt_trajectory'] = gt_trajectory

        # Past frames gt depth
        if self.cfg.LIFT.GT_DEPTH:
            depths = batch['depths']
            depth_labels = depths[:, :self.model.receptive_field, :, ::self.model.encoder_downsample,
                           ::self.model.encoder_downsample]
            depth_labels = depth_labels / self.cfg.LIFT.D_BOUND[2]
            d_min = self.cfg.LIFT.D_BOUND[0] / self.cfg.LIFT.D_BOUND[2]
            d_max = self.cfg.LIFT.D_BOUND[1] / self.cfg.LIFT.D_BOUND[2] - 1
            depth_labels = torch.clamp(depth_labels, d_min, d_max) - d_min
            depth_labels = depth_labels.long().contiguous()
            labels['depths'] = depth_labels

        # Warp labels to present's reference frame
        # segmentation_labels_past = cumulative_warp_features(
        #     segmentation_labels[:, :self.model.receptive_field].float(),
        #     future_egomotion[:, :self.model.receptive_field],
        #     mode='nearest', spatial_extent=self.spatial_extent,
        # ).long().contiguous()[:, :-1]
        # segmentation_labels = cumulative_warp_features_reverse(
        #     segmentation_labels[:, (self.model.receptive_field - 1):].float(),
        #     future_egomotion[:, (self.model.receptive_field - 1):],
        #     mode='nearest', spatial_extent=self.spatial_extent,
        # ).long().contiguous()
        # labels['segmentation'] = torch.cat([segmentation_labels_past, segmentation_labels], dim=1)
        labels['segmentation'] = segmentation_labels.long().contiguous()
        labels['segmentation_occ'] = (1 - segmentation_labels * \
             invisible_mask).long().contiguous().float()
        future_inputs.append(labels['segmentation'][:, \
            self.model.receptive_field - 1:] * labels['segmentation_occ'][:, \
            self.model.receptive_field - 1:])
        if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            # pedestrian_labels = batch['pedestrian']
            # pedestrian_labels_past = cumulative_warp_features(
            #     pedestrian_labels[:, :self.model.receptive_field].float(),
            #     future_egomotion[:, :self.model.receptive_field],
            #     mode='nearest', spatial_extent=self.spatial_extent,
            # ).long().contiguous()[:, :-1]
            # pedestrian_labels = cumulative_warp_features_reverse(
            #     pedestrian_labels[:, (self.model.receptive_field - 1):].float(),
            #     future_egomotion[:, (self.model.receptive_field - 1):],
            #     mode='nearest', spatial_extent=self.spatial_extent,
            # ).long().contiguous()
            # labels['pedestrian'] = torch.cat([pedestrian_labels_past, pedestrian_labels], dim=1)
            if self.cfg.USE_DETECTION:
                labels['pedestrian'] = batch['pedestrian_dec'].long().contiguous()
            else:
                labels['pedestrian'] = batch['pedestrian'].long().contiguous()
            labels['pedestrian_occ'] = (1 - batch['pedestrian'] * \
                invisible_mask).long().contiguous().float()
            labels['obstacle'] = batch['obstacle'].long().contiguous()
            labels['obstacle_occ'] = (1 - batch['obstacle'] * \
                invisible_mask).long().contiguous().float()
            future_inputs.append(labels['pedestrian'][:, \
                self.model.receptive_field - 1:] * labels['pedestrian_occ'][:, \
                self.model.receptive_field - 1:])
        labels['command'], labels['target_point'] = batch['command'].long(), batch['target_point'].float()
        # Warp instance labels to present's reference frame
        # if self.cfg.INSTANCE_SEG.ENABLED:
        #     gt_instance = batch['instance']
        #     instance_center_labels = batch['centerness']
        #     instance_offset_labels = batch['offset']
        #     gt_instance_past = cumulative_warp_features(
        #         gt_instance[:, :self.model.receptive_field].float().unsqueeze(2),
        #         future_egomotion[:, :self.model.receptive_field],
        #         mode='nearest', spatial_extent=self.spatial_extent,
        #     ).long().contiguous()[:, :-1, 0]
        #     gt_instance = cumulative_warp_features_reverse(
        #         gt_instance[:, (self.model.receptive_field - 1):].float().unsqueeze(2),
        #         future_egomotion[:, (self.model.receptive_field - 1):],
        #         mode='nearest', spatial_extent=self.spatial_extent,
        #     ).long().contiguous()[:, :, 0]
        #     labels['instance'] = torch.cat([gt_instance_past, gt_instance], dim=1)

        #     instance_center_labels_past = cumulative_warp_features(
        #         instance_center_labels[:, :self.model.receptive_field],
        #         future_egomotion[:, :self.model.receptive_field],
        #         mode='nearest', spatial_extent=self.spatial_extent,
        #     ).contiguous()[:, :-1]
        #     instance_center_labels = cumulative_warp_features_reverse(
        #         instance_center_labels[:, (self.model.receptive_field - 1):],
        #         future_egomotion[:, (self.model.receptive_field - 1):],
        #         mode='nearest', spatial_extent=self.spatial_extent,
        #     ).contiguous()
        #     labels['centerness'] = torch.cat([instance_center_labels_past, instance_center_labels], dim=1)

        #     instance_offset_labels_past = cumulative_warp_features(
        #         instance_offset_labels[:, :self.model.receptive_field],
        #         future_egomotion[:, :self.model.receptive_field],
        #         mode='nearest', spatial_extent=self.spatial_extent,
        #     ).contiguous()[:, :-1]
        #     instance_offset_labels = cumulative_warp_features_reverse(
        #         instance_offset_labels[:, (self.model.receptive_field - 1):],
        #         future_egomotion[:, (self.model.receptive_field - 1):],
        #         mode='nearest', spatial_extent=self.spatial_extent,
        #     ).contiguous()
        #     labels['offset'] = torch.cat([instance_offset_labels_past, instance_offset_labels], dim=1)

        # if self.cfg.INSTANCE_FLOW.ENABLED:
        #     instance_flow_labels = batch['flow']
        #     instance_flow_labels_past = cumulative_warp_features(
        #         instance_flow_labels[:, :self.model.receptive_field],
        #         future_egomotion[:, :self.model.receptive_field],
        #         mode='nearest', spatial_extent=self.spatial_extent,
        #     ).contiguous()[:, :-1]
        #     instance_flow_labels = cumulative_warp_features_reverse(
        #         instance_flow_labels[:, (self.model.receptive_field - 1):],
        #         future_egomotion[:, (self.model.receptive_field - 1):],
        #         mode='nearest', spatial_extent=self.spatial_extent,
        #     ).contiguous()
        #     labels['flow'] = torch.cat([instance_flow_labels_past, instance_flow_labels], dim=1)
        future_inputs = torch.cat(future_inputs, dim=2)
        stop_future_labels = torch.cat([stop_labels[:, self.model.receptive_field - 1:].clone(),
            stop_labels[:, self.model.receptive_field - 1:].clone()], dim=2)
        future_inputs = torch.cat([future_inputs, \
            hdmap_labels[:, self.model.receptive_field - 1:].clone(),
            stop_future_labels], dim=1).contiguous()
        return labels, future_inputs

    def visualise(self, labels, output, batch_idx, prefix='train'):
        visualisation_video = visualise_output(labels, output, self.cfg)
        name = f'{prefix}_outputs'
        if prefix == 'train':
            self.logger.experiment.add_video(name, visualisation_video, global_step=self.training_step_count, fps=2)
        else:
            name = f'{name}-{self.training_step_count}'
            self.logger.experiment.add_video(name, visualisation_video, \
                global_step=batch_idx, fps=2)

    def training_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, True)
        self.training_step_count += 1
        for key, value in loss.items():
            if 'uncertainty' in key:
                self.logger.experiment.add_scalar('step_train_' + key, value, global_step=self.training_step_count)
            else:
                self.logger.experiment.add_scalar('step_train_loss_' + key, value, global_step=self.training_step_count)
        if self.training_step_count % self.cfg.VIS_INTERVAL == 0:
            self.visualise(labels, output, batch_idx, prefix='train')
        return loss['total']

    def validation_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, False)
        scores = self.metric_vehicle_val.compute()
        self.log('epoch_val_all_seg_iou_vehicle', scores[1])
        # self.log('step_predicted_traj_x', output['selected_traj'][0, -1, 0])
        # self.log('step_target_traj_x', labels['gt_trajectory'][0, -1, 0])
        # self.log('step_predicted_traj_y', output['selected_traj'][0, -1, 1])
        # self.log('step_target_traj_y', labels['gt_trajectory'][0, -1, 1])

        if batch_idx % 200 == 0:
            self.visualise(labels, output, batch_idx, prefix='val')

    def shared_epoch_end(self, step_outputs, is_train):
        if not is_train:
            scores = self.metric_vehicle_val.compute()
            self.logger.experiment.add_scalar('epoch_val_all_seg_iou_vehicle', scores[1],
                                              global_step=self.training_step_count)
            self.metric_vehicle_val.reset()

            if self.cfg.SEMANTIC_SEG.SUBGOAL.ENABLED:
                scores = self.metric_subgoal_val.compute()
                self.logger.experiment.add_scalar('epoch_val_all_subgoal_iou', scores[1],
                                                  global_step=self.training_step_count)
                self.metric_subgoal_val.reset()

            if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
                scores = self.metric_pedestrian_val.compute()
                self.logger.experiment.add_scalar('epoch_val_all_seg_iou_pedestrian', scores[1],
                                                  global_step=self.training_step_count)
                self.metric_pedestrian_val.reset()

            if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
                # for i, name in enumerate(self.hdmap_class):
                #     scores = self.metric_hdmap_val[i].compute()
                #     self.logger.experiment.add_scalar('epoch_val_hdmap_iou_' + name, scores[1],
                #                                       global_step=self.training_step_count)
                #     self.metric_hdmap_val[i].reset()
                if self.cfg.SEMANTIC_SEG.LANE.ENABLED:
                    scores = self.metric_lane_val.compute()
                    self.logger.experiment.add_scalar('epoch_val_lane_iou', scores[1],
                                                      global_step=self.training_step_count)
                    self.metric_lane_val.reset()
                if self.cfg.SEMANTIC_SEG.AREA.ENABLED:
                    scores = self.metric_area_val.compute()
                    self.logger.experiment.add_scalar('epoch_val_area_iou', scores[1],
                                                      global_step=self.training_step_count)
                    self.metric_area_val.reset()
            
            if self.cfg.SEMANTIC_SEG.OBSTACLE.ENABLED:
                scores = self.metric_obstacle_val.compute()
                self.logger.experiment.add_scalar('epoch_val_obstacle_iou', scores[1],
                                                    global_step=self.training_step_count)
                self.metric_obstacle_val.reset()

            if self.cfg.SEMANTIC_SEG.STOP.ENABLED:
                scores = self.metric_stop_val.compute()
                self.logger.experiment.add_scalar('epoch_val_stop_iou', scores[1],
                                                    global_step=self.training_step_count)
                self.metric_stop_val.reset()

            if self.cfg.INSTANCE_SEG.ENABLED:
                scores = self.metric_panoptic_val.compute()
                for key, value in scores.items():
                    self.logger.experiment.add_scalar(f'epoch_val_all_ins_{key}_vehicle', value[1].item(),
                                                      global_step=self.training_step_count)
                self.metric_panoptic_val.reset()

            if self.cfg.PLANNING.ENABLED:
                scores = self.metric_planning_val.compute()
                for key, value in scores.items():
                    self.logger.experiment.add_scalar('epoch_val_plan_' + key, value.mean(),
                                                      global_step=self.training_step_count)
                self.metric_planning_val.reset()

        # self.logger.experiment.add_scalar('epoch_segmentation_weight',
        #                                   1 / (2 * torch.exp(self.model.segmentation_weight)),
        #                                   global_step=self.training_step_count)
        # if self.cfg.LIFT.GT_DEPTH:
        #     self.logger.experiment.add_scalar('epoch_depths_weight', 1 / (2 * torch.exp(self.model.depths_weight)),
        #                                       global_step=self.training_step_count)
        # if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
        #     self.logger.experiment.add_scalar('epoch_pedestrian_weight',
        #                                       1 / (2 * torch.exp(self.model.pedestrian_weight)),
        #                                       global_step=self.training_step_count)
        # if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        #     self.logger.experiment.add_scalar('epoch_hdmap_weight', 1 / (2 * torch.exp(self.model.hdmap_weight)),
        #                                       global_step=self.training_step_count)
        # if self.cfg.INSTANCE_SEG.ENABLED:
        #     self.logger.experiment.add_scalar('epoch_centerness_weight',
        #                                       1 / (2 * torch.exp(self.model.centerness_weight)),
        #                                       global_step=self.training_step_count)
        #     self.logger.experiment.add_scalar('epoch_offset_weight', 1 / (2 * torch.exp(self.model.offset_weight)),
        #                                       global_step=self.training_step_count)
        # if self.cfg.INSTANCE_FLOW.ENABLED:
        #     self.logger.experiment.add_scalar('epoch_flow_weight', 1 / (2 * torch.exp(self.model.flow_weight)),
        #                                       global_step=self.training_step_count)
        # if self.cfg.PLANNING.ENABLED:
        #     self.logger.experiment.add_scalar('epoch_planning_weight', 1 / (2 * torch.exp(self.model.planning_weight)),
        #                                       global_step=self.training_step_count)

    def training_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, True)

    def validation_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, False)

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = torch.optim.Adam(
            params, lr=self.cfg.OPTIMIZER.LR, weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
        )

        return optimizer
