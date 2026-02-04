import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models.vgg import vgg16


##########################################################################
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.target_down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)

    def forward(self, x, y):
        _, _, x_kw, x_kh = x.shape
        _, _, y_kw, y_kh = y.shape
        if x_kw == y_kw:
            diff = x - y
        else:
            diff = x - self.target_down(y)
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss
##########################################################################


##########################################################################
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()
        self.target_down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        _, _, x_kw, x_kh = x.shape
        _, _, y_kw, y_kh = y.shape
        if x_kw == y_kw:
            loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        else:
            loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(self.target_down(y)))
        # loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
##########################################################################
