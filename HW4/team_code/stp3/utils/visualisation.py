import numpy as np
import torch
import matplotlib.pylab
import matplotlib.pyplot as plt
from stp3.utils.tools import gen_dx_bx


from stp3.utils.instance import predict_instance_segmentation_and_trajectories

DEFAULT_COLORMAP = matplotlib.pylab.cm.jet
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
    "aqua": (0, 255, 255),
    "pink": (255, 182, 193),
    "orange": (255, 165, 0),
    "brown": (150, 105, 25)
}

def flow_to_image(flow: np.ndarray, autoscale: bool = False) -> np.ndarray:
    """
    Applies colour map to flow which should be a 2 channel image tensor HxWx2. Returns a HxWx3 numpy image
    Code adapted from: https://github.com/liruoteng/FlowNet/blob/master/models/flownet/scripts/flowlib.py
    """
    u = flow[0, :, :]
    v = flow[1, :, :]

    # Convert to polar coordinates
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = np.max(rad)

    # Normalise flow maps
    if autoscale:
        u /= maxrad + np.finfo(float).eps
        v /= maxrad + np.finfo(float).eps

    # visualise flow with cmap
    return np.uint8(compute_color(u, v) * 255)


def _normalise(image: np.ndarray) -> np.ndarray:
    lower = np.min(image)
    delta = np.max(image) - lower
    if delta == 0:
        delta = 1
    image = (image.astype(np.float32) - lower) / delta
    return image


def apply_colour_map(
    image: np.ndarray, cmap: matplotlib.colors.LinearSegmentedColormap = DEFAULT_COLORMAP, autoscale: bool = False
) -> np.ndarray:
    """
    Applies a colour map to the given 1 or 2 channel numpy image. if 2 channel, must be 2xHxW.
    Returns a HxWx3 numpy image
    """
    if image.ndim == 2 or (image.ndim == 3 and image.shape[0] == 1):
        if image.ndim == 3:
            image = image[0]
        # grayscale scalar image
        if autoscale:
            image = _normalise(image)
        return cmap(image)[:, :, :3]
    if image.shape[0] == 2:
        # 2 dimensional UV
        return flow_to_image(image, autoscale=autoscale)
    if image.shape[0] == 3:
        # normalise rgb channels
        if autoscale:
            image = _normalise(image)
        return np.transpose(image, axes=[1, 2, 0])
    raise Exception('Image must be 1, 2 or 3 channel to convert to colour_map (CxHxW)')


def heatmap_image(
    image: np.ndarray, cmap: matplotlib.colors.LinearSegmentedColormap = DEFAULT_COLORMAP, autoscale: bool = True
) -> np.ndarray:
    """Colorize an 1 or 2 channel image with a colourmap."""
    if not issubclass(image.dtype.type, np.floating):
        raise ValueError(f"Expected a ndarray of float type, but got dtype {image.dtype}")
    if not (image.ndim == 2 or (image.ndim == 3 and image.shape[0] in [1, 2])):
        raise ValueError(f"Expected a ndarray of shape [H, W] or [1, H, W] or [2, H, W], but got shape {image.shape}")
    heatmap_np = apply_colour_map(image, cmap=cmap, autoscale=autoscale)
    heatmap_np = np.uint8(heatmap_np * 255)
    return heatmap_np


def compute_color(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    assert u.shape == v.shape
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nan_mask = np.isnan(u) | np.isnan(v)
    u[nan_mask] = 0
    v[nan_mask] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    f_k = (a + 1) / 2 * (ncols - 1) + 1
    k_0 = np.floor(f_k).astype(int)
    k_1 = k_0 + 1
    k_1[k_1 == ncols + 1] = 1
    f = f_k - k_0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k_0 - 1] / 255
        col1 = tmp[k_1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = col * (1 - nan_mask)

    return img


def make_color_wheel() -> np.ndarray:
    """
    Create colour wheel.
    Code adapted from https://github.com/liruoteng/FlowNet/blob/master/models/flownet/scripts/flowlib.py
    """
    red_yellow = 15
    yellow_green = 6
    green_cyan = 4
    cyan_blue = 11
    blue_magenta = 13
    magenta_red = 6

    ncols = red_yellow + yellow_green + green_cyan + cyan_blue + blue_magenta + magenta_red
    colorwheel = np.zeros([ncols, 3])

    col = 0

    # red_yellow
    colorwheel[0:red_yellow, 0] = 255
    colorwheel[0:red_yellow, 1] = np.transpose(np.floor(255 * np.arange(0, red_yellow) / red_yellow))
    col += red_yellow

    # yellow_green
    colorwheel[col : col + yellow_green, 0] = 255 - np.transpose(
        np.floor(255 * np.arange(0, yellow_green) / yellow_green)
    )
    colorwheel[col : col + yellow_green, 1] = 255
    col += yellow_green

    # green_cyan
    colorwheel[col : col + green_cyan, 1] = 255
    colorwheel[col : col + green_cyan, 2] = np.transpose(np.floor(255 * np.arange(0, green_cyan) / green_cyan))
    col += green_cyan

    # cyan_blue
    colorwheel[col : col + cyan_blue, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, cyan_blue) / cyan_blue))
    colorwheel[col : col + cyan_blue, 2] = 255
    col += cyan_blue

    # blue_magenta
    colorwheel[col : col + blue_magenta, 2] = 255
    colorwheel[col : col + blue_magenta, 0] = np.transpose(np.floor(255 * np.arange(0, blue_magenta) / blue_magenta))
    col += +blue_magenta

    # magenta_red
    colorwheel[col : col + magenta_red, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, magenta_red) / magenta_red))
    colorwheel[col : col + magenta_red, 0] = 255

    return colorwheel


def make_contour(img, colour=[0, 0, 0], double_line=False):
    h, w = img.shape[:2]
    out = img.copy()
    # Vertical lines
    out[np.arange(h), np.repeat(0, h)] = colour
    out[np.arange(h), np.repeat(w - 1, h)] = colour

    # Horizontal lines
    out[np.repeat(0, w), np.arange(w)] = colour
    out[np.repeat(h - 1, w), np.arange(w)] = colour

    if double_line:
        out[np.arange(h), np.repeat(1, h)] = colour
        out[np.arange(h), np.repeat(w - 2, h)] = colour

        # Horizontal lines
        out[np.repeat(1, w), np.arange(w)] = colour
        out[np.repeat(h - 2, w), np.arange(w)] = colour
    return out


def plot_instance_map(instance_image, instance_map, instance_colours=None, bg_image=None):
    if isinstance(instance_image, torch.Tensor):
        instance_image = instance_image.cpu().numpy()
    assert isinstance(instance_image, np.ndarray)
    if instance_colours is None:
        instance_colours = generate_instance_colours(instance_map)
    if len(instance_image.shape) > 2:
        instance_image = instance_image.reshape((instance_image.shape[-2], instance_image.shape[-1]))

    if bg_image is None:
        plot_image = 255 * np.ones((instance_image.shape[0], instance_image.shape[1], 3), dtype=np.uint8)
    else:
        plot_image = bg_image

    for key, value in instance_colours.items():
        plot_image[instance_image == key] = value

    return plot_image

@ torch.no_grad()
def visualise_output(labels, output, cfg):
    for key in output:
        if key in labels:
            if labels[key] is not None and output[key] is not None and torch.is_tensor(labels[key]) and torch.is_tensor(output[key]):
                labels[key], output[key] = labels[key].detach().clone(), output[key].detach().clone()
                # if torch.is_tensor(labels[key]) and torch.is_tensor(output[key]):
                #     labels[key][..., :cfg.TRUNCATE] = 0.
                #     output[key][..., :cfg.TRUNCATE] = 0.
                #     labels[key][..., -cfg.TRUNCATE-1:] = 0.
                #     output[key][..., -cfg.TRUNCATE-1:] = 0.
    visible_mask = 1 - labels['visible_mask'].detach().clone()
    # visible_mask[..., :cfg.TRUNCATE] = 0.
    # visible_mask[..., -cfg.TRUNCATE-1:] = 0.
    visible_mask = visible_mask.squeeze(2).cpu().numpy()
    for obj_type in ['pedestrian', 'segmentation', 'stop', 'obstacle']:
        if f'{obj_type}_occ' in labels and output[obj_type] is not None:
            occ_mask = labels[f'{obj_type}_occ'][:, :labels[obj_type].shape[1]]
            labels[obj_type] = labels[obj_type] * occ_mask
            output[obj_type] = output[obj_type][:, :labels[obj_type].shape[1]] * occ_mask[:, :output[obj_type].shape[1]]

    nonzero_indices = lambda arr: arr != 0
    
    if cfg.INSTANCE_SEG.ENABLED:
        consistent_instance_seg = predict_instance_segmentation_and_trajectories(
            output, compute_matched_centers=False
        )

    sequence_length = labels['segmentation'].shape[1]
    b = 0
    video = []
    if sequence_length != cfg.TIME_RECEPTIVE_FIELD and cfg.PRETRAINED.FREEZE:
        start = cfg.TIME_RECEPTIVE_FIELD - 1
    else:
        start = 0
    for t in range(start, sequence_length):
        out_t = []
        rt = min(cfg.TIME_RECEPTIVE_FIELD - 1, t)

        # Ground truth
        label_plots = []
        semantic_seg = labels['segmentation'].squeeze(2).cpu().numpy()
        semantic_plot = np.zeros((*semantic_seg[b, t].shape, 3))
        semantic_plot[nonzero_indices(semantic_seg[b, t])] = colors['purple']
        
        if cfg.SEMANTIC_SEG.SUBGOAL.ENABLED:
            subgoal_seg = labels['subgoal'].squeeze(2).cpu().numpy()
            semantic_plot[nonzero_indices(subgoal_seg[b, 0])] = colors['blue']
            semantic_plot[nonzero_indices(subgoal_seg[b, -1])] = colors['red']

        if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            pedestrian_seg = labels['pedestrian'].squeeze(2).cpu().numpy()
            semantic_plot[nonzero_indices(pedestrian_seg[b, t])] = colors['teal']
        label_plots.append(semantic_plot)

        # if cfg.INSTANCE_SEG.ENABLED:
        #     unique_ids = torch.unique(labels['instance'][b, t]).cpu().numpy()[1:]
        #     instance_map = dict(zip(unique_ids, unique_ids))
        #     instance_plot = plot_instance_map(labels['instance'][b, t].cpu(), instance_map)
        #     instance_plot = make_contour(instance_plot)
        #     label_plots.append(instance_plot)

        #     center_plot = heatmap_image(labels['centerness'][b, t, 0].cpu().numpy())
        #     center_plot = make_contour(center_plot)
        #     label_plots.append(center_plot)

        #     offset_plot = labels['offset'][b, t].cpu().numpy()
        #     offset_plot[:, semantic_seg[b, t] != 1] = 0
        #     offset_plot = flow_to_image(offset_plot)
        #     offset_plot = make_contour(offset_plot)
        #     label_plots.append(offset_plot)

        # if cfg.INSTANCE_FLOW.ENABLED:
        #     future_flow_plot = labels['flow'][b, t].cpu().numpy()
        #     future_flow_plot[:, semantic_seg[b, t] != 1] = 0
        #     future_flow_plot = flow_to_image(future_flow_plot)
        #     future_flow_plot = make_contour(future_flow_plot)
        #     label_plots.append(future_flow_plot)

        # planning_plot = plot_planning(labels['hdmap'][b], labels['gt_trajectory'][b], cfg)
        # planning_plot = make_contour(planning_plot)
        # label_plots.append(planning_plot)
        area_seg = labels['area'].squeeze(2).cpu().numpy()
        # area_seg = labels['hdmap'][:, 1].cpu().numpy()
        area_plot = np.zeros((*area_seg[b, rt].shape, 3))
        area_plot[nonzero_indices(area_seg[b, rt])] = colors['white']
        label_plots.append(area_plot)

        hdmap_plot = area_plot.copy()
        if cfg.SEMANTIC_SEG.LANE.ENABLED:
            lane_seg = labels['lane'].squeeze(2).cpu().numpy()
            hdmap_plot[nonzero_indices(lane_seg[b, rt])] = colors['teal']
        if cfg.SEMANTIC_SEG.STOP.ENABLED:
            stop_seg = labels['stop'].squeeze(2).cpu().numpy()
            hdmap_plot[nonzero_indices(stop_seg[b, rt])] = colors['orange']
        if cfg.SEMANTIC_SEG.OBSTACLE.ENABLED:
            obs_seg = labels['obstacle'].squeeze(2).cpu().numpy()
            hdmap_plot[nonzero_indices(obs_seg[b, rt])] = colors['brown']
        label_plots.append(hdmap_plot)

        total_plot = hdmap_plot.copy()
        total_plot[nonzero_indices(semantic_seg[b, t])] = colors['purple']
        total_plot[nonzero_indices(pedestrian_seg[b, t])] = colors['teal']
        if cfg.SEMANTIC_SEG.SUBGOAL.ENABLED:
            total_plot[nonzero_indices(subgoal_seg[b, 0])] = colors['blue']
            total_plot[nonzero_indices(subgoal_seg[b, -1])] = colors['red']
        w, h, _ = total_plot.shape
        total_plot[w // 2 - 6: w // 2 + 6, 
                h // 2 - 3: h //2 + 3] = colors["blue"]
        total_plot[nonzero_indices(visible_mask[b, t])] += 80
        total_plot = np.clip(total_plot, a_min=0, a_max=255)
        label_plots.append(total_plot)

        out_t.append(np.concatenate(label_plots, axis=1))
        
        # Predictions
        prediction_plots = []
        semantic_seg = output['segmentation'].argmax(dim=2).detach().cpu().numpy()
        semantic_plot = np.zeros((*semantic_seg[b, t].shape, 3))
        semantic_plot[nonzero_indices(semantic_seg[b, t])] = colors['purple']
        
        if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            if output['pedestrian'].shape[2] > 1:
                pedestrian_seg = output['pedestrian'].argmax(dim=2).detach().cpu().numpy()
            else:
                pedestrian_seg = output['pedestrian'].detach().squeeze(2).cpu().numpy()
            semantic_plot[nonzero_indices(pedestrian_seg[b, t])] = colors['teal']
        
        if cfg.SEMANTIC_SEG.SUBGOAL.ENABLED:
            subgoal_seg = output['subgoal'].detach().squeeze(2).cpu().numpy()
            semantic_plot[nonzero_indices(subgoal_seg[b, 0])] = colors['blue']
            semantic_plot[nonzero_indices(subgoal_seg[b, -1])] = colors['red']
        prediction_plots.append(semantic_plot)

        # if cfg.INSTANCE_SEG.ENABLED:
        #     unique_ids = torch.unique(consistent_instance_seg[b, t]).cpu().numpy()[1:]
        #     instance_map = dict(zip(unique_ids, unique_ids))
        #     instance_plot = plot_instance_map(consistent_instance_seg[b, t].cpu(), instance_map)
        #     instance_plot = make_contour(instance_plot)
        #     prediction_plots.append(instance_plot)

        #     center_plot = heatmap_image(output['instance_center'][b, t, 0].detach().cpu().numpy())
        #     center_plot = make_contour(center_plot)
        #     prediction_plots.append(center_plot)

        #     offset_plot = output['instance_offset'][b, t].detach().cpu().numpy()
        #     offset_plot[:, semantic_seg[b, t] != 1] = 0
        #     offset_plot = flow_to_image(offset_plot)
        #     offset_plot = make_contour(offset_plot)
        #     prediction_plots.append(offset_plot)

        # if cfg.INSTANCE_FLOW.ENABLED:
        #     future_flow_plot = output['instance_flow'][b, t].detach().cpu().numpy()
        #     future_flow_plot[:, semantic_seg[b, t] != 1] = 0
        #     future_flow_plot = flow_to_image(future_flow_plot)
        #     future_flow_plot = make_contour(future_flow_plot)
        #     prediction_plots.append(future_flow_plot)

        if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            # output['hdmap'][:, :2] = torch.nn.functional.softmax(output['hdmap'][:, :2], dim=1)
            # hdmap1 = (output['hdmap'][:, 1:2] > 0.7).long()
            # hdmap2 = output['hdmap'][:, 2:].argmax(dim=1, keepdims=True)
            # hdmap = torch.cat([hdmap1, hdmap2], dim=1)
            # planning_plot = plot_planning(hdmap[b], output['selected_traj'][b], cfg)
            # planning_plot = make_contour(planning_plot)
            # prediction_plots.append(planning_plot)
            # output['hdmap'][:, :2] = torch.nn.functional.softmax(output['hdmap'][:, :2], dim=1)
            # hdmap1 = (output['hdmap'][:, 1:2] > 0.7).long()[:, 0]
            # hdmap2 = output['hdmap'][:, 2:].argmax(dim=1)
            # area_seg = hdmap2.cpu().numpy()
            area_seg = output['area'].argmax(dim=2).detach().cpu().numpy()
            area_plot = np.zeros((*area_seg[b, rt].shape, 3))
            area_plot[nonzero_indices(area_seg[b, rt])] = colors['white']
            prediction_plots.append(area_plot)

            hdmap_plot = area_plot.copy()
            if cfg.SEMANTIC_SEG.LANE.ENABLED:
                lane_seg = output['lane'].argmax(dim=2).detach().cpu().numpy()
                hdmap_plot[nonzero_indices(lane_seg[b, rt])] = colors['teal']
            if cfg.SEMANTIC_SEG.OBSTACLE.ENABLED:
                obs_seg = output['obstacle'].argmax(dim=2).detach().cpu().numpy()
                hdmap_plot[nonzero_indices(obs_seg[b, rt])] = colors['brown']
            if cfg.SEMANTIC_SEG.STOP.ENABLED:
                if output['stop'].shape[2] > 1:
                    stop_seg = output['stop'].argmax(dim=2).detach().cpu().numpy()
                else:
                    stop_seg = output['stop'].detach().squeeze(2).cpu().numpy()            
                hdmap_plot[nonzero_indices(stop_seg[b, rt])] = colors['orange']
            prediction_plots.append(hdmap_plot)

            total_plot = hdmap_plot.copy()
            total_plot[nonzero_indices(semantic_seg[b, t])] = colors['purple']
            total_plot[nonzero_indices(pedestrian_seg[b, t])] = colors['teal']
            if cfg.SEMANTIC_SEG.SUBGOAL.ENABLED:
                total_plot[nonzero_indices(subgoal_seg[b, 0])] = colors['blue']
                total_plot[nonzero_indices(subgoal_seg[b, -1])] = colors['red']
        
            w, h, _ = total_plot.shape
            total_plot[w // 2 - 6: w // 2 + 6, 
                h // 2 - 3: h //2 + 3] = colors["blue"]
            total_plot[nonzero_indices(visible_mask[b, t])] += 80
            total_plot = np.clip(total_plot, a_min=0, a_max=255)
            prediction_plots.append(total_plot)
        
        out_t.append(np.concatenate(prediction_plots, axis=1))
        out_t = np.concatenate(out_t, axis=0)
        # Shape (C, H, W)
        out_t = out_t.transpose((2, 0, 1))

        video.append(out_t)

    # Shape (B, T, C, H, W)
    video = np.stack(video)[None].astype(np.uint8)
    return video


def convert_figure_numpy(figure):
    """ Convert figure to numpy image """
    figure_np = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    figure_np = figure_np.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    return figure_np

def plot_planning(hd_map, traj, cfg):
    '''
    hd_map: torch.tensor (2, 200, 200)
    traj: torch.tensor (n_future, 2)
    '''
    if isinstance(hd_map, torch.Tensor):
        hd_map = hd_map.detach().cpu().numpy()
    if isinstance(traj, torch.Tensor):
        traj = traj.detach().cpu().numpy()

    h, w = hd_map.shape[-2:]
    fig = plt.figure(figsize=(w/100, h/100))

    dx, bx, _ = gen_dx_bx(cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND)
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    map = np.ones((h, w, 3))
    yx = np.nonzero(hd_map[1])
    c = np.array([1.00, 0.50, 0.31])
    map[yx[0], yx[1], :] = c
    yx = np.nonzero(hd_map[0])
    c = np.array([0., 0., 0.2])
    map[yx[0], yx[1], :] = c
    plt.imshow(map, alpha=0.2)
    plt.axis('off')

    plt.xlim((w, 0))
    plt.ylim((0, h))
    W = cfg.EGO.WIDTH
    H = cfg.EGO.HEIGHT
    pts = np.array([
        [-H / 2. + 0.5, W / 2.],
        [H / 2. + 0.5, W / 2.],
        [H / 2. + 0.5, -W / 2.],
        [-H / 2. + 0.5, -W / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')

    gt = (traj[:, :2] - bx) / dx
    plt.plot(gt[:, 0], gt[:, 1])

    plt.draw()
    figure_numpy = convert_figure_numpy(fig)
    plt.close()


    return figure_numpy

def generate_instance_colours(instance_map):
    # Most distinct 22 colors (kelly colors from https://stackoverflow.com/questions/470690/how-to-automatically-generate
    # -n-distinct-colors)
    # plus some colours from AD40k
    INSTANCE_COLOURS = np.asarray([
        [0, 0, 0],
        [255, 179, 0],
        [128, 62, 117],
        [255, 104, 0],
        [166, 189, 215],
        [193, 0, 32],
        [206, 162, 98],
        [129, 112, 102],
        [0, 125, 52],
        [246, 118, 142],
        [0, 83, 138],
        [255, 122, 92],
        [83, 55, 122],
        [255, 142, 0],
        [179, 40, 81],
        [244, 200, 0],
        [127, 24, 13],
        [147, 170, 0],
        [89, 51, 21],
        [241, 58, 19],
        [35, 44, 22],
        [112, 224, 255],
        [70, 184, 160],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [0, 255, 235],
        [255, 0, 235],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 255, 204],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [255, 214, 0],
        [25, 194, 194],
        [92, 0, 255],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
    ])

    return {instance_id: INSTANCE_COLOURS[global_instance_id % len(INSTANCE_COLOURS)] for
            instance_id, global_instance_id in instance_map.items()
            }
