import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
from stp3.models.center_net import CenterNetHead
from stp3.layers.convolutions import UpsamplingAdd, DeepLabHead


class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, n_present, n_hdmap, predict_gate, use_detection):
        super().__init__()
        self.perceive_area = predict_gate['perceive_area']
        self.perceive_lane = predict_gate['perceive_lane']
        self.perceive_stop = predict_gate['perceive_stop']
        self.perceive_traffic = predict_gate['perceive_traffic']
        self.perceive_pedestrian = predict_gate['perceive_pedestrian']
        self.perceive_obstacle = predict_gate['perceive_obstacle']
        self.predict_instance = predict_gate['predict_instance']
        self.predict_future_flow = predict_gate['predict_future_flow']
        self.planning = predict_gate['planning']
        self.use_detection = use_detection

        self.n_classes = n_classes
        self.n_present = n_present
        if self.predict_instance is False and self.predict_future_flow is True:
            raise ValueError('flow cannot be True when not predicting instance')

        backbone = resnet18(pretrained=False, zero_init_residual=True)

        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, self.n_classes, kernel_size=1, padding=0),
        )

        if self.perceive_pedestrian:
            if self.use_detection:
                self.pedestrian_head = CenterNetHead(shared_out_channels, 4, 0.4, [3])
            else:
                self.pedestrian_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, self.n_classes, kernel_size=1, padding=0),
            )

        if self.perceive_area:
            # self.hdmap_head = nn.Sequential(
            #     nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            #     nn.BatchNorm2d(shared_out_channels),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(shared_out_channels, 2 * n_hdmap, kernel_size=1, padding=0),
            # )
            self.area_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

        if self.perceive_obstacle:
            self.obs_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

        if self.perceive_traffic:
            self.traffic_head1 = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=3, padding=1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(inplace=True),
            )
            
            self.traffic_head2 = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2)
            )
            # self.traffic_head1 = nn.Sequential(
            #     nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1),
            #     nn.InstanceNorm2d(shared_out_channels),
            #     nn.LeakyReLU(inplace=True),
            #     nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1),
            #     nn.InstanceNorm2d(shared_out_channels),
            #     nn.LeakyReLU(inplace=True),
            # )
            # self.traffic_head2 = nn.Sequential(
            #     nn.Linear(shared_out_channels, shared_out_channels),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(shared_out_channels, 2)
            # )

        if self.perceive_stop:
            # if self.use_detection:
            #     self.stop_head = CenterNetHead(shared_out_channels, 4, 0.4, [2, 6])
            # else:
            self.stop_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

        if self.perceive_lane:
            self.lane_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

        self.perceive_subgoal = predict_gate['perceive_subgoal']
        if self.perceive_subgoal:
            self.command_embedding = nn.Embedding(6, 128)  
            self.target_embedding = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 128),
                nn.ReLU(inplace=True)
            )
            self.subgoal_mlp = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024)
            )
            self.subgoal_cnn = nn.Sequential(
                nn.ConvTranspose2d(1, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            self.subgoal_head = nn.Sequential(
                nn.Conv2d(shared_out_channels * 3 + 32, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 5, kernel_size=1, padding=0),
            )
            # self.subgoal_head = CenterNetHead(shared_out_channels, 4, 0.4, [5], output_factor=5)

        if self.predict_instance:
            self.instance_offset_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )
            self.instance_center_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid(),
            )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

        if self.planning:
            self.costvolume_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            )

    def forward(self, x, hidden, lidar=None, target_point=None, command=None):
        traffic_output = None
        if self.perceive_traffic:
            b, s, n, c, h, w = hidden.shape
            hidden = hidden.view(b * s * n, c, h, w)
            hidden = self.traffic_head1(hidden)
            hidden = hidden.view(b, s, n, -1, h, w)
            hidden = hidden.sum((2, 4, 5))
            traffic_output = self.traffic_head2(hidden).view(b, s, -1)
            
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
            
        # (H, W)
        skip_x = {'1': x}

        # (H/2, W/2)
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        skip_x['2'] = x

        # (H/4 , W/4)
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)  # (b*s, 256, 25, 25)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])
        segmentation_output = self.segmentation_head(x)
        pedestrian_output = stop_output = None
        pedestrian_detection = stop_detection = None
        
        subgoal_output = None
        if self.perceive_subgoal and target_point is not None:
            command = self.command_embedding(command).squeeze(1)
            target_point = self.target_embedding(target_point)
            cond = torch.cat([command, target_point], dim=-1)
            cond = self.subgoal_mlp(cond).view(-1, 32, 32).unsqueeze(1)
            cond = self.subgoal_cnn(cond)
            subgoal_inputs = torch.cat([x.view(b, -1, h, w), cond], dim=1)
            subgoal_heatmap = self.subgoal_head(subgoal_inputs).sigmoid()
            subgoal_output = (subgoal_heatmap >= 0.45).long()
            # subgoal_heatmap, subgoal_offset = self.subgoal_head(subgoal_inputs)
            # subgoal_heatmap = subgoal_heatmap.view(-1, 1, *subgoal_heatmap.shape[2:])
            # subgoal_offset = subgoal_offset.view(-1, 2, *subgoal_offset.shape[2:])
            # subgoal_output = self.subgoal_head.decode_heatmap(subgoal_heatmap, subgoal_offset)
            # subgoal_detection = {'center_heatmap_pred': subgoal_heatmap.view(b, 5, *subgoal_heatmap.shape[1:]),
            #                         'offset_pred': subgoal_offset.view(b, 5, *subgoal_offset.shape[1:])}

        if self.use_detection:
            if self.perceive_pedestrian:
                pedestrian_heatmap, pedestrian_offset = self.pedestrian_head(x)
                pedestrian_output = self.pedestrian_head.decode_heatmap(pedestrian_heatmap, pedestrian_offset)
                pedestrian_output = pedestrian_output.view(b, s, *pedestrian_output.shape[1:])
                pedestrian_detection = {'center_heatmap_pred': pedestrian_heatmap.view(b, s, *pedestrian_heatmap.shape[1:]),
                                    'offset_pred': pedestrian_offset.view(b, s, *pedestrian_offset.shape[1:])}
            # if self.perceive_stop:
            #     stop_heatmap, stop_offset = self.stop_head(x)
            #     stop_output = self.stop_head.decode_heatmap(stop_heatmap, stop_offset)
            #     stop_output = stop_output.view(b, s, *stop_output.shape[1:])
            #     stop_detection = {'center_heatmap_pred': stop_heatmap.view(b, s, *stop_heatmap.shape[1:])[:, :self.n_present],
            #                         'offset_pred': stop_offset.view(b, s, *stop_offset.shape[1:])[:, :self.n_present]}
        else:
            if self.perceive_pedestrian:
                pedestrian_output = self.pedestrian_head(x) if self.perceive_pedestrian else None
                pedestrian_output = pedestrian_output.view(b, s, *pedestrian_output.shape[1:])
        if self.perceive_stop:
            stop_output = self.stop_head(x) if self.perceive_area else None
            stop_output = stop_output.view(b, s, *stop_output.shape[1:])

        # hdmap_output = self.hdmap_head(x.view(b, s, *x.shape[1:])[:,self.n_present-1]) if self.perceive_hdmap else None
        lane_output = self.lane_head(x) if self.perceive_lane else None
        area_output = self.area_head(x) if self.perceive_area else None
        obs_output = self.obs_head(x) if self.perceive_obstacle else None
        instance_center_output = self.instance_center_head(x) if self.predict_instance else None
        instance_offset_output = self.instance_offset_head(x) if self.predict_instance else None
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None
        costvolume = self.costvolume_head(x).squeeze(1) if self.planning else None
        
        # traffic_output = None
        # if self.perceive_traffic:
        #     traffic_output = self.traffic_head1(x)
        #     traffic_output = traffic_output.sum((-1, -2))
        #     traffic_output = self.traffic_head2(traffic_output)

        results = {
            'segmentation': segmentation_output.view(b, s, *segmentation_output.shape[1:]),
            'pedestrian': pedestrian_output if pedestrian_output is not None else None,
            'obstacle': obs_output.view(b, s, *obs_output.shape[1:]) if obs_output is not None else None,
            # 'hdmap' : hdmap_output,
            'lane' : lane_output.view(b, s, *lane_output.shape[1:]) if lane_output is not None else None,
            'area' : area_output.view(b, s, *area_output.shape[1:]) if area_output is not None else None,
            'stop' : stop_output if stop_output is not None else None,
            'instance_center': instance_center_output.view(b, s, *instance_center_output.shape[1:])
            if instance_center_output is not None else None,
            'instance_offset': instance_offset_output.view(b, s, *instance_offset_output.shape[1:])
            if instance_offset_output is not None else None,
            'instance_flow': instance_future_output.view(b, s, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
            'costvolume': costvolume.view(b, s, *costvolume.shape[1:])
            if costvolume is not None else None,
            'traffic': traffic_output if traffic_output is not None else None,
            'subgoal': subgoal_output if subgoal_output is not None else None,
        }
        if self.use_detection:
            if pedestrian_detection is not None:
                results['pedestrian_detection'] = pedestrian_detection
            # if stop_detection is not None:
            #     results['stop_detection'] = stop_detection
        if self.perceive_subgoal:
            results['subgoal_heat'] = subgoal_heatmap
        return results
