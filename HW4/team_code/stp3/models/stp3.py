import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from stp3.models.encoder import Encoder
from stp3.models.temporal_model import TemporalModelIdentity, TemporalModel
from stp3.models.distributions import DistributionModule
from stp3.models.future_prediction import FuturePrediction
from stp3.models.decoder import Decoder
from stp3.models.planning_model import Planning
from stp3.utils.network import pack_sequence_dim, unpack_sequence_dim, set_bn_momentum
from stp3.utils.geometry import calculate_birds_eye_view_parameters, VoxelsSumming, pose_vec2mat

class STP3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.visualize = False

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.encoder_downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE
        self.encoder_out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS

        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape
        self.discount = self.cfg.LIFT.DISCOUNT

        if self.cfg.TIME_RECEPTIVE_FIELD == 1:
            assert self.cfg.MODEL.TEMPORAL_MODEL.NAME == 'identity'

        # temporal block
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD
        self.n_future = self.cfg.N_FUTURE_FRAMES
        self.latent_dim = self.cfg.MODEL.DISTRIBUTION.LATENT_DIM
        self.temporal_coef = torch.eye(self.receptive_field)
        for i in range(self.receptive_field):
            for j in range(self.receptive_field):
                self.temporal_coef[i, j] = self.discount**abs(i - j)
        self.temporal_coef = torch.flip(self.temporal_coef, [0, 1])
        self.temporal_coef = self.temporal_coef.view(1, 3, 3, 1, 1, 1)
        self.temporal_coef = nn.Parameter(self.temporal_coef, requires_grad=False)

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())

        # Encoder
        self.encoder = Encoder(cfg=self.cfg.MODEL.ENCODER, D=self.depth_channels)

        # Temporal model
        temporal_in_channels = self.encoder_out_channels
        if self.cfg.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE:
            temporal_in_channels += 6
        if self.cfg.MODEL.INPUT_LIDAR:
            temporal_in_channels += 2
        if self.cfg.MODEL.TEMPORAL_MODEL.NAME == 'identity':
            self.temporal_model = TemporalModelIdentity(temporal_in_channels, self.receptive_field)
        elif cfg.MODEL.TEMPORAL_MODEL.NAME == 'temporal_block':
            self.temporal_model = TemporalModel(
                temporal_in_channels,
                self.receptive_field,
                input_shape=self.bev_size,
                start_out_channels=self.cfg.MODEL.TEMPORAL_MODEL.START_OUT_CHANNELS,
                extra_in_channels=self.cfg.MODEL.TEMPORAL_MODEL.EXTRA_IN_CHANNELS,
                n_spatial_layers_between_temporal_layers=self.cfg.MODEL.TEMPORAL_MODEL.INBETWEEN_LAYERS,
                use_pyramid_pooling=self.cfg.MODEL.TEMPORAL_MODEL.PYRAMID_POOLING,
            )
        else:
            raise NotImplementedError(f'Temporal module {self.cfg.MODEL.TEMPORAL_MODEL.NAME}.')

        self.future_pred_in_channels = self.temporal_model.out_channels
        if self.cfg.MODEL.INPUT_LIDAR:
            self.future_pred_in_channels += 2
        if self.n_future > 0:
            # probabilistic sampling
            if self.cfg.PROBABILISTIC.ENABLED:
                # Distribution networks
                self.present_distribution = DistributionModule(
                    self.future_pred_in_channels,
                    self.latent_dim,
                    method=self.cfg.PROBABILISTIC.METHOD
                )

            if self.cfg.PROBABILISTIC.GT_FUTURE:
                future_distribution_in_channels = (self.future_pred_in_channels
                                                    + self.n_future * 2 + 4
                                                    )
                self.future_distribution = DistributionModule(
                    future_distribution_in_channels,
                    self.latent_dim,
                    method=self.cfg.PROBABILISTIC.METHOD
                )

            # Future prediction
            self.future_prediction = FuturePrediction(
                in_channels=self.future_pred_in_channels,
                latent_dim=self.latent_dim,
                n_future=self.n_future,
                mixture=self.cfg.MODEL.FUTURE_PRED.MIXTURE,
                n_gru_blocks=self.cfg.MODEL.FUTURE_PRED.N_GRU_BLOCKS,
                n_res_layers=self.cfg.MODEL.FUTURE_PRED.N_RES_LAYERS,
            )

        # Decoder
        self.decoder = Decoder(
            in_channels=self.future_pred_in_channels,
            n_classes=len(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS),
            n_present=self.receptive_field,
            n_hdmap=len(self.cfg.SEMANTIC_SEG.HDMAP.ELEMENTS),
            predict_gate = {
                'perceive_area': self.cfg.SEMANTIC_SEG.AREA.ENABLED,
                'perceive_stop': self.cfg.SEMANTIC_SEG.STOP.ENABLED,
                'perceive_traffic': self.cfg.SEMANTIC_SEG.TRAFFIC.ENABLED,
                'perceive_obstacle': self.cfg.SEMANTIC_SEG.OBSTACLE.ENABLED,
                'perceive_lane': self.cfg.SEMANTIC_SEG.LANE.ENABLED,
                'perceive_pedestrian': self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED,
                'predict_instance': self.cfg.INSTANCE_SEG.ENABLED,
                'predict_future_flow': self.cfg.INSTANCE_FLOW.ENABLED,
                'planning': self.cfg.PLANNING.ENABLED,
                'perceive_subgoal': self.cfg.SEMANTIC_SEG.SUBGOAL.ENABLED
            },
            use_detection = self.cfg.USE_DETECTION
        )

        # Cost function
        # Carla 128, Nuscenes 256
        if self.cfg.PLANNING.ENABLED:
            self.planning = Planning(cfg, self.encoder_out_channels, 6, gru_state_size=self.cfg.PLANNING.GRU_STATE_SIZE)

        set_bn_momentum(self, self.cfg.MODEL.BN_MOMENTUM)

    def create_frustum(self):
        # Create grid in image plane
        h, w = self.cfg.IMAGE.FINAL_DIM
        downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample

        # Depth grid
        depth_grid = torch.arange(*self.cfg.LIFT.D_BOUND, dtype=torch.float) # [2.0, 50.0, 1.0]
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        # x and y grids
        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
        # containing data points in the image: left-right, top-bottom, depth
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)

        return nn.Parameter(frustum, requires_grad=False)

    def forward(self, image, intrinsics, extrinsics, future_egomotion, \
            affine_mats, future_inputs, lidar, is_train, depths, target_point=None, command=None):
        output = {}

        # Only process features from the past and present
        image = image[:, :self.receptive_field].contiguous()
        intrinsics = intrinsics[:, :self.receptive_field].contiguous()
        extrinsics = extrinsics[:, :self.receptive_field].contiguous()
        affine_mats = affine_mats[:, :self.receptive_field].contiguous()
        future_egomotion = future_egomotion[:, :self.receptive_field].contiguous()
        lidar = lidar[:, :self.receptive_field].contiguous()
        if future_inputs is not None:
            future_inputs = future_inputs.contiguous()

        # Lifting features and project to bird's-eye view
        x, depth, hidden, cam_front = self.calculate_birds_eye_view_features(image, intrinsics, extrinsics, affine_mats, depths) # (3,3,64,200,200)
        output = {**output, 'depth_prediction': depth, 'cam_front':cam_front}

        if self.cfg.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE:
            b, s, c = future_egomotion.shape
            h, w = x.shape[-2:]
            future_egomotions_spatial = future_egomotion.view(b, s, c, 1, 1).expand(b, s, c, h, w)
            # at time 0, no egomotion so feed zero vector
            future_egomotions_spatial = torch.cat([torch.zeros_like(future_egomotions_spatial[:, :1]),
                                                   future_egomotions_spatial[:, :(self.receptive_field-1)]], dim=1)
            x = torch.cat([x, future_egomotions_spatial], dim=-3)
        if self.cfg.MODEL.INPUT_LIDAR:
            x = torch.cat([x, lidar], dim=2)

        #  Temporal model
        states = self.temporal_model(x)
        if self.cfg.MODEL.INPUT_LIDAR:
            states = torch.cat([states, lidar], dim=2)
        else:
            lidar = None
        hidden = unpack_sequence_dim(hidden, image.shape[0], image.shape[1])
        hidden = hidden.permute(0, 1, 2, 4, 5, 3, 6).contiguous()
        b, s, n, h, w, c, d = hidden.shape
        hidden = hidden.view(b, s, n, h, w, -1)
        hidden = hidden[:, :, [0, 1, 3]].permute(0, 1, 2, 5, 3, 4).contiguous()
        # b, s, n, d, h, w, c
        if self.n_future > 0:
            present_state = states[:, -1:].contiguous()

            b, _, c, h, w = present_state.shape

            if self.cfg.PROBABILISTIC.ENABLED:
                sample, output_distribution = self.distribution_forward(
                    present_state, future_inputs, is_train,
                    min_log_sigma=self.cfg.MODEL.DISTRIBUTION.MIN_LOG_SIGMA,
                    max_log_sigma=self.cfg.MODEL.DISTRIBUTION.MAX_LOG_SIGMA,
                )
                output = {**output, 'output_distribution': output_distribution}
                future_prediction_input = sample
            else:
                future_prediction_input = present_state.new_zeros(b, 1, self.latent_dim, h, w)

            # Recursively predict states
            states = self.future_prediction(future_prediction_input, states)

            # Predict BEV outputs
            bev_output = self.decoder(states, hidden, target_point, command)

        else:
            # Perceive BEV outputs
            bev_output = self.decoder(states, hidden, target_point, command)

        output = {**output, **bev_output}

        return output

    def get_geometry(self, intrinsics, extrinsics):
        """Calculate the (x, y, z) 3D position of the features.
        """
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Camera to ego reference frame
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)

        return points

    def encoder_forward(self, x, cam_front_index=1):
        # batch, n_cameras, channels, height, width
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w) # (9 * 6, 3, 224, 480)
        x, depth = self.encoder(x) # (9 * 6, 64, 28, 60)
        if self.cfg.PLANNING.ENABLED:
            cam_front = x.view(b, n, *x.shape[1:])[:, cam_front_index]
        else:
            cam_front = None

        if self.cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION:
            depth_prob = depth.softmax(dim=1)
            x = depth_prob.unsqueeze(1) * x.unsqueeze(2)  # outer product depth and features
        else:
            x = x.unsqueeze(2).repeat(1, 1, self.depth_channels, 1, 1)

        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2) # channel dimension
        depth = depth.view(b, n, *depth.shape[1:])

        return x, depth, cam_front

    def projection_to_birds_eye_view(self, x, geometry):
        """ Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200"""
        # batch, n_cameras, depth, height, width, channels
        batch, n, d, h, w, c = x.shape
        output = torch.zeros(
            (batch, c, self.bev_dimension[0], self.bev_dimension[1]), dtype=torch.float, device=x.device
        )

        # Number of 3D points
        N = n * d * h * w
        for b in range(batch):
            # flatten x
            x_b = x[b].reshape(N, c)

            # Convert positions to integer indices
            geometry_b = ((geometry[b] - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
            geometry_b = geometry_b.view(N, 3).long()

            # Mask out points that are outside the considered spatial extent.
            mask = (
                    (geometry_b[:, 0] >= 0)
                    & (geometry_b[:, 0] < self.bev_dimension[0])
                    & (geometry_b[:, 1] >= 0)
                    & (geometry_b[:, 1] < self.bev_dimension[1])
                    & (geometry_b[:, 2] >= 0)
                    & (geometry_b[:, 2] < self.bev_dimension[2])
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]

            # Sort tensors so that those within the same voxel are consecutives.
            ranks = (
                    geometry_b[:, 0] * (self.bev_dimension[1] * self.bev_dimension[2])
                    + geometry_b[:, 1] * (self.bev_dimension[2])
                    + geometry_b[:, 2]
            )
            ranks_indices = ranks.argsort()
            x_b, geometry_b, ranks = x_b[ranks_indices], geometry_b[ranks_indices], ranks[ranks_indices]

            # Project to bird's-eye view by summing voxels.
            x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)

            bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                      device=x_b.device)
            bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b

            # Put channel in second position and remove z dimension
            bev_feature = bev_feature.permute((0, 3, 1, 2))
            bev_feature = bev_feature.squeeze(0)

            output[b] = bev_feature

        return output

    @ torch.no_grad()
    def map_img_to_bev(self, x, geometry):
        batch, n, d, h, w, c = x.shape
        output = torch.zeros(
            (batch, c, self.bev_dimension[0], self.bev_dimension[1]), dtype=torch.float, device=x.device
        )

        # Number of 3D points
        N = n * d * h * w
        for b in range(batch):
            # flatten x
            x_b = x[b].reshape(N, c)

            # Convert positions to integer indices
            geometry_b = ((geometry[b] - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
            geometry_b = geometry_b.view(N, 3).long()

            # Mask out points that are outside the considered spatial extent.
            mask = (
                    (geometry_b[:, 0] >= 0)
                    & (geometry_b[:, 0] < self.bev_dimension[0])
                    & (geometry_b[:, 1] >= 0)
                    & (geometry_b[:, 1] < self.bev_dimension[1])
                    & (geometry_b[:, 2] >= 0)
                    & (geometry_b[:, 2] < self.bev_dimension[2])
                    & ((x_b[:, 0] != 0)
                    | (x_b[:, 1] != 0)
                    | (x_b[:, 2] != 0))
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]

            bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                      device=x_b.device)
            bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b

            # Put channel in second position and remove z dimension
            bev_feature = bev_feature.permute((0, 3, 1, 2))
            bev_feature = bev_feature.squeeze(0)

            output[b] = bev_feature

        return output

    def warp_features(self, x, affine_mats):
        """
        Applies affine transformations to feature maps using the provided affine matrices.
        
        Parameters:
            x (torch.Tensor): The input feature map tensor, with shape (N, C, H, W).
            affine_mats (torch.Tensor): A tensor of affine transformation matrices you created by create_affine_mat() function.

        Returns:
            torch.Tensor: The warped BEV map tensor with the same shape as the input `x`.
        """
        # TODO_2-2: Implement the warp_features() function
        N, C, H, W = x.shape
        grid = F.affine_grid(affine_mats, size=(N, C, H, W), align_corners=False)
        new_x = F.grid_sample(
            x, 
        	grid, 
        	mode='bilinear',  # interpolation
        	padding_mode='zeros',  # Pad with zeros
        	align_corners=False
   	    )
        
        return new_x

    def calculate_birds_eye_view_features(self, x, intrinsics, extrinsics, affine_mats, depths):
        """
        This function processes input image and camera's parameters to compute the bird's-eye view features.
        Please follow the steps below to implement this function:
        1. Encode the input features using the encoder_forward() function.
        2. Project the encoded features to the bird's-eye view using the projection_to_birds_eye_view() function.
        3. After get each frame's BEV feature, warp them to ego's current coordinate using the warp_features() function.
        4. With self.temporal_coef, Apply the temporal fusion to the warped features.
        """
        b, s, n, c, h, w = x.shape
        # Reshape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)
        affine_mats = pack_sequence_dim(affine_mats)
        geometry = self.get_geometry(intrinsics, extrinsics)

        # TODO_3: Complete the calculate_birds_eye_view_features() function
        encoded_features, depth, cam_front = self.encoder_forward(x) 
        bev_features = self.projection_to_birds_eye_view(encoded_features, geometry)  
        warped_features = self.warp_features(bev_features, affine_mats)
        warped_features = unpack_sequence_dim(warped_features, b, s)  
    
    	# Temporal fusion (according to ST-P3 paper: x_t = b_t + \Sigma^t−1_i=1 α_i × ̃x_t−i)
        temporal_fused_features = warped_features.clone()  
        for t in range(1, s):
            for i in range(1, t):  # Loop over past frames
                temporal_fused_features += (self.temporal_coef[:, t, i] ** i) * temporal_fused_features[:, t - i]            
            temporal_fused_features[:, t] += warped_features[:, t]

        geometry = unpack_sequence_dim(geometry, b, s)
        depth = unpack_sequence_dim(depth, b, s)
        cam_front = unpack_sequence_dim(cam_front, b, s)[:,-1] if cam_front is not None else None
        
        return temporal_fused_features, depth, encoded_features, cam_front

    def distribution_forward(self, present_features, future_inputs, is_train, min_log_sigma, max_log_sigma):
        """
        Parameters
        ----------
            present_features: 5-D output from dynamics module with shape (b, 1, c, h, w)

        Returns
        -------
            sample: sample taken from present/future distribution, broadcast to shape (b, s, latent_dim, h, w)
        """
        b, s, _, h, w = present_features.size()
        assert s == 1

        def get_mu_sigma(mu_log_sigma):
            mu = mu_log_sigma[:, :, :self.latent_dim]
            log_sigma = mu_log_sigma[:, :, self.latent_dim:2*self.latent_dim]
            log_sigma = torch.clamp(log_sigma, min_log_sigma, max_log_sigma)
            sigma = torch.exp(log_sigma)
            if future_inputs is not None:
                gaussian_noise = torch.randn((b, s, self.latent_dim), device=present_features.device)
            else:
                gaussian_noise = torch.zeros((b, s, self.latent_dim), device=present_features.device)
            sample = mu + sigma * gaussian_noise
            return mu, log_sigma, sample


        if self.cfg.PROBABILISTIC.METHOD == 'GAUSSIAN':
            mu_log_sigma = self.present_distribution(present_features)
            present_mu, present_log_sigma, present_sample = get_mu_sigma(mu_log_sigma)
            output_distribution = {
                'present_mu': present_mu,
                'present_log_sigma': present_log_sigma,
            }

            if future_inputs is not None:
                # Concatenate future labels to z_t
                future_features = future_inputs[:, 1:].contiguous().view(b, 1, -1, h, w)
                future_features = torch.cat([present_features, future_features], dim=2)
                mu_log_sigma = self.future_distribution(future_features)
                future_mu, future_log_sigma, future_sample = get_mu_sigma(mu_log_sigma)
                output_distribution.update({
                    'future_mu': future_mu,
                    'future_log_sigma': future_log_sigma,
                })
                sample = future_sample
            elif is_train:        
                sample = present_sample
            else:
                sample = present_mu

            # Spatially broadcast sample to the dimensions of present_features
            sample = sample.view(b, s, self.latent_dim, 1, 1).expand(b, s, self.latent_dim, h, w)

        elif self.cfg.PROBABILISTIC.METHOD == "BERNOULLI":
            present_log_prob = self.present_distribution(present_features)
            if self.training:
                bernoulli_noise = torch.randn((b, self.latent_dim, h, w), device=present_features.device)
            else:
                bernoulli_noise = torch.zeros((b, self.latent_dim, h, w), device=present_features.device)
            sample = torch.exp(present_log_prob) + bernoulli_noise

            sample = sample.view(b, s, self.latent_dim, h, w)


        elif self.cfg.PROBABILISTIC.METHOD == 'MIXGAUSSIAN':
            mu_log_sigma = self.present_distribution(present_features)
            present_mu1, present_log_sigma1, present_sample1 = get_mu_sigma(mu_log_sigma[:, :, :2*self.latent_dim])
            present_mu2, present_log_sigma2, present_sample2 = get_mu_sigma(mu_log_sigma[:, :, 2 * self.latent_dim : 4 * self.latent_dim])
            present_mu3, present_log_sigma3, present_sample3 = get_mu_sigma(mu_log_sigma[:, :, 4 * self.latent_dim : 6 * self.latent_dim])
            coefficient = mu_log_sigma[:, :, 6 * self.latent_dim:]
            coefficient = torch.softmax(coefficient, dim=-1)
            sample = present_sample1 * coefficient[:,:,0:1] + \
                     present_sample2 * coefficient[:,:,1:2] + \
                     present_sample3 * coefficient[:,:,2:3]

            # Spatially broadcast sample to the dimensions of present_features
            sample = sample.view(b, s, self.latent_dim, 1, 1).expand(b, s, self.latent_dim, h, w)

        else:
            raise NotImplementedError

        return sample, output_distribution

    def select_best_traj(self, trajs, cost_volume, lane_divider, semantic_pred, k=1):
        '''
        trajs: torch.Tensor (B, N, n_future, 3)
        fuser_feature: torch.Tensor (B, n_present, fuser_channel, 200, 200)
        hd_map: torch.Tensor(B, 5, 200, 200)
        semantic_pred: torch.Tensor(B, n_future, 200, 200)
        '''
        sm_cost_fc, sm_cost_fo = self.cost_function(cost_volume, trajs[:,:,:,:2], semantic_pred, lane_divider)

        CS = sm_cost_fc + sm_cost_fo.sum(dim=-1)
        CC, KK = torch.topk(CS, k, dim=-1, largest=False)

        ii = torch.arange(len(trajs))
        select_traj = trajs[ii[:,None], KK].squeeze(1) # (B, n_future, 3)

        return select_traj
