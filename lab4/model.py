import torch
import torch.nn as nn

import torch.optim


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()

        self.append(nn.BatchNorm2d(num_maps_in))  # TODO CHECK
        self.append(nn.ReLU())
        self.append(nn.Conv2d(num_maps_in, num_maps_out, bias=bias, kernel_size=k))


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        self.relu_conv1 = _BNReluConv(input_channels, input_channels)
        self.relu_conv2 = _BNReluConv(input_channels, input_channels)
        self.relu_conv3 = _BNReluConv(input_channels, emb_size)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2)

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        x = self.relu_conv1(img)
        x = self.max_pool(x)
        x = self.relu_conv2(x)
        x = self.max_pool(x)
        x = self.relu_conv3(x)
        x = self.max_pool(x)

        return x.unsqueeze(0)

    def loss(self, anchor, positive, negative, margin=1):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)

        d_a_p = torch.norm(a_x - p_x)
        d_a_n = torch.norm(a_x - n_x)

        loss = torch.max(d_a_p - d_a_n + margin, 0)
        return loss