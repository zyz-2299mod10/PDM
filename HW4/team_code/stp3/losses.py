import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class SpatialRegressionLoss(nn.Module):
    def __init__(self, norm, ignore_index=255, future_discount=1.0):
        super(SpatialRegressionLoss, self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index
        self.future_discount = future_discount

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target, n_present=3, occ_mask=None):
        assert len(prediction.shape) == 5, 'Must be a 5D tensor'
        # ignore_index is the same across all channels
        mask = target[:, :, :1] != self.ignore_index
        if mask.sum() == 0:
            return prediction.new_zeros(1)[0].float()
        loss = self.loss_fn(prediction, target, reduction='none')
        if occ_mask is not None:
            loss = loss * occ_mask

        # Sum channel dimension
        loss = torch.sum(loss, dim=-3, keepdim=True)

        seq_len = loss.shape[1]
        assert seq_len >= n_present
        future_len = seq_len - n_present
        future_discounts = self.future_discount ** torch.arange(1, future_len+1, device=loss.device, dtype=loss.dtype)
        discounts = torch.cat([torch.ones(n_present, device=loss.device, dtype=loss.dtype), future_discounts], dim=0)
        discounts = discounts.view(1, seq_len, 1, 1, 1)
        loss = loss * discounts

        return loss[mask].mean()


class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False, top_k_ratio=1.0, future_discount=1.0):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount

    def forward(self, prediction, target, occ_mask=None):
        if target.shape[-3] != 1:
            raise ValueError('segmentation label must be an index-label with channel dimension = 1.')
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)
        loss = F.cross_entropy(
            prediction,
            target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights.to(target.device),
        )
        if occ_mask is not None:
            occ_mask = occ_mask.view(b * s, h, w)
            loss = loss * occ_mask

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)

class HDmapLoss(nn.Module):
    def __init__(self, class_weights, training_weights, use_top_k, top_k_ratio, ignore_index=255):
        super(HDmapLoss, self).__init__()
        self.class_weights = class_weights
        self.training_weights = training_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio

    def forward(self, prediction, target):
        loss = 0
        for i in range(target.shape[-3]):
            cur_target = target[:, i]
            b, h, w = cur_target.shape
            cur_prediction = prediction[:, 2*i:2*(i+1)]
            cur_loss = F.cross_entropy(
                cur_prediction,
                cur_target,
                ignore_index=self.ignore_index,
                reduction='none',
                weight=self.class_weights[i].to(target.device),
            )

            cur_loss = cur_loss.view(b, -1)
            if self.use_top_k[i]:
                k = int(self.top_k_ratio[i] * cur_loss.shape[1])
                cur_loss, _ = torch.sort(cur_loss, dim=1, descending=True)
                cur_loss = cur_loss[:, :k]
            loss += torch.mean(cur_loss) * self.training_weights[i]
        return loss

class DepthLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=255):
        super(DepthLoss, self).__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        b, s, n, d, h, w = prediction.shape
        prediction = prediction.view(b*s*n, d, h, w)
        target = target.view(b*s*n, h, w)
        loss = F.cross_entropy(
            prediction,
            target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights
        )
        return torch.mean(loss)

class ProbabilisticLoss(nn.Module):
    def __init__(self, method):
        super(ProbabilisticLoss, self).__init__()
        self.method = method

    def kl_div(self, present_mu, present_log_sigma, future_mu, future_log_sigma):
        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
                present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                2 * var_present)
        )

        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))
        return kl_loss

    def forward(self, output):
        if self.method == 'GAUSSIAN':
            present_mu = output['present_mu']
            present_log_sigma = output['present_log_sigma']
            future_mu = output['future_mu']
            future_log_sigma = output['future_log_sigma']

            kl_loss = self.kl_div(present_mu, present_log_sigma, future_mu, future_log_sigma)
        elif self.method == 'MIXGAUSSIAN':
            present_mu = output['present_mu']
            present_log_sigma = output['present_log_sigma']
            future_mu = output['future_mu']
            future_log_sigma = output['future_log_sigma']

            kl_loss = 0
            for i in range(len(present_mu)):
                kl_loss += self.kl_div(present_mu[i], present_log_sigma[i], future_mu[i], future_log_sigma[i])
        elif self.method == 'BERNOULLI':
            present_log_prob = output['present_log_prob']
            future_log_prob = output['future_log_prob']

            kl_loss = F.kl_div(present_log_prob, future_log_prob, reduction='batchmean', log_target=True)
        else:
            raise NotImplementedError
        return kl_loss


class BEVDetectionLoss(nn.Module):
    def __init__(self):
        super(BEVDetectionLoss, self).__init__()
        # We use none reduction because we weight each pixel according to the number of bounding boxes.
        self.loss_offset = nn.L1Loss(reduction='none')

    def gaussian_focal_loss(self, pred, gaussian_target, 
            alpha=2.0, gamma=4.0, eps=1e-12, reduction='mean'):
        """ Adapted from mmdetection
        Args:
            pred (torch.Tensor): The prediction.
            gaussian_target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 2.0.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 4.0.
        """
        pos_weights = gaussian_target.eq(1) * 10
        neg_weights = (1 - gaussian_target).pow(gamma)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
        neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
        loss = pos_loss + neg_loss

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        # All other reductions will be no reduction.
        return loss

    def forward(self, center_heatmap_pred, offset_pred, 
            center_heatmap_target, offset_target, pixel_mask):
        """
        Compute losses of the head.

        Args:
            center_heatmap_preds (Tensor): center predict heatmaps for all levels with shape (B, num_classes, H, W).
            wh_preds (Tensor): wh predicts for all levels with shape (B, 2, H, W).
            offset_preds (Tensor): offset predicts for all levels with shape (B, 2, H, W).

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        loss_center_heatmap = self.gaussian_focal_loss(center_heatmap_pred, center_heatmap_target,
                                                    reduction='mean') * 1e2
        # For the other predictions this value is 1 so it is omitted.
        loss_offset = (self.loss_offset(offset_pred, offset_target) * pixel_mask).sum() / \
            (offset_pred.shape[0] * offset_pred.shape[1]) * 1e-1
        return loss_center_heatmap, loss_offset

class FocalLoss(nn.Module):
    
    def __init__(self, class_weights=None, 
                 gamma=2., reduction='mean', **kwargs):
        nn.Module.__init__(self)
        self.weight = class_weights
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor, occ_mask=None):
        b, s, c, h, w = input_tensor.shape
        input_tensor = input_tensor.view(b * s, c, h, w)
        target_tensor = target_tensor.view(b * s, h, w)
        
        log_prob = F.log_softmax(input_tensor, dim=1)
        prob = torch.exp(log_prob)
        loss = F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight.to(target_tensor.device),
            reduction = 'none'
        )
        if occ_mask is not None:
            occ_mask = occ_mask.view(b * s, h, w)
            loss = loss * occ_mask
        return loss.mean() * 1e2


class GaussianFocalLoss(nn.Module):
    
    def __init__(self, weight=2,
        alpha=2.0, gamma=4.0, eps=1e-12, reduction='mean', **kwargs):
        nn.Module.__init__(self)
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, pred, gaussian_target):
        """ Adapted from mmdetection
        Args:
            pred (torch.Tensor): The prediction.
            gaussian_target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 2.0.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 4.0.
        """
        pos_weights = gaussian_target.eq(1) * self.weight
        neg_weights = (1 - gaussian_target).pow(self.gamma)
        pos_loss = -(pred + self.eps).log() * (1 - pred).pow(self.alpha) * pos_weights
        neg_loss = -(1 - pred + self.eps).log() * pred.pow(self.alpha) * neg_weights
        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # All other reductions will be no reduction.
        return loss * 1e2
