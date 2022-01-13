import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.ce = nn.CrossEntropyLoss()
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)

        # cosine.acos_()
        # cosine[index] += m_hot
        # cosine.cos_().mul_(self.s)

        sine = torch.sqrt((1. - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        cosine = m_hot * phi + (1. - m_hot) * cosine
        cosine = self.s * cosine

        loss = self.ce(cosine, label)
        return loss


class AmFaceLoss(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(AmFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.ce = nn.CrossEntropyLoss()

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)

        cosine[index] -= m_hot
        cosine.mul_(self.s)
        loss = self.ce(cosine, label)
        return loss


class SVAmLoss(nn.Module):
    def __init__(self, s=64, m=0.5, t=1.2):
        super().__init__()
        self.s = s
        self.m = m
        self.t = t

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], 1.0)

        phi_theta = cosine - self.m

        fm = torch.sum(phi_theta * m_hot, dim=-1)
        fm_expand = torch.unsqueeze(fm, dim=1)
        Ix = fm_expand - cosine  # [N, n_class]
        Ix = Ix < 0
        Ix = Ix.float()
        h = torch.exp(self.s * (self.t - 1.) * (cosine + 1.0) * Ix)
        reversed_target = 1 - m_hot
        sum_sv = reversed_target * h * torch.exp(self.s * cosine)
        sum_sv = torch.sum(sum_sv, dim=1)
        loss = (-1) * (self.s * fm - torch.log(torch.exp(self.s * fm) + sum_sv))
        loss = torch.mean(loss)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class SamSmoothingCrossEntropy(nn.Module):
    """
    Sam loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(SamSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, gold):
        n_class = pred.size(1)
        one_hot = torch.full_like(pred, fill_value=self.smoothing / (n_class - 1))
        one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=self.confidence)
        log_prob = F.log_softmax(pred, dim=1)
        loss = F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)
        return loss.mean()


"""
def sva_am_loss(y_true,y_pred,s,m,t):

    cos_theta = y_pred
    phi_theta = cos_theta - m
    fm = tf.reduce_sum(phi_theta * y_true, axis=-1)  # [N],the f(m,theta) in SV-x-loss.

    fm_expand = tf.expand_dims(fm, axis=1)
    I = fm_expand - cos_theta  # [N,n_class]
    I = I < 0
    I = tf.cast(I, tf.float32)
    h = tf.exp(s * (t - 1) * (cos_theta + 1.0) * I)
    reversed_target = 1 - y_true
    sum_sv = reversed_target * h * tf.exp(s * cos_theta)  # [N,num_classes]
    sum_sv = tf.reduce_sum(sum_sv, axis=1)
    loss = (-1) * (s * fm - tf.math.log(tf.exp(s * fm) + sum_sv))  # [N]
    loss = tf.reduce_mean(loss)
    return loss


"""

if __name__ == '__main__':
    import numpy as np

    cosine = torch.rand(8, 100)
    label = np.array([2, 3, 4, 1, 6, 3, 1, 4], dtype=np.int64)
    label = torch.from_numpy(label)
    criteria = AmFaceLoss()
    loss = criteria(cosine, label)
    print(loss)
