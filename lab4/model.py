import torch
import torch.nn as nn

import torch.optim


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()

        self.num_maps_in = num_maps_in
        self.num_maps_out = num_maps_out
        self.k = k
        self.bias = bias

        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU())
        self.append(nn.Conv2d(num_maps_in, num_maps_out, bias=bias, kernel_size=k))


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size

        self.relu_conv1 = _BNReluConv(input_channels, emb_size, k=3)
        self.relu_conv2 = _BNReluConv(emb_size, emb_size, k=3)
        self.relu_conv3 = _BNReluConv(emb_size, emb_size, k=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        x = self.relu_conv1(img)
        x = self.max_pool(x)
        x = self.relu_conv2(x)
        x = self.max_pool(x)
        x = self.relu_conv3(x)
        x = self.avg_pool(x)

        return x.reshape((img.size(dim=0), self.emb_size))

    def loss(self, anchor, positive, negative, margin=1):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)

        d_a_p = torch.linalg.norm(a_x - p_x, dim=1)
        d_a_n = torch.linalg.norm(a_x - n_x, dim=1)

        loss = torch.mean(torch.maximum(d_a_p - d_a_n + margin, torch.tensor(0)))

        return loss


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        feats = img.reshape((img.size(dim=0), img.size(dim=2) * (img.size(dim=3))))
        return feats

    def loss(self, anchor, positive, negative, margin=1):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)

        d_a_p = torch.linalg.norm(a_x - p_x, dim=1)
        d_a_n = torch.linalg.norm(a_x - n_x, dim=1)

        loss = torch.mean(torch.maximum(d_a_p - d_a_n + margin, torch.tensor(0)))

        return loss
