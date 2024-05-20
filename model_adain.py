import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

'''
Adaptive Instance Normalization (AdaIN) 구현
'''

def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        self.loss = F.mse_loss(input * self.weight, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature, weight):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach() * weight
        self.weight = weight
        self.loss = F.mse_loss(self.target, self.target)

    @staticmethod
    def gram_matrix(input):
        batch_size, f_map_num, h, w = input.size()
        features = input.view(batch_size * f_map_num, h * w)
        G = torch.mm(features, features.t())
        return G.div(batch_size * f_map_num * h * w)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G * self.weight, self.target)
        return input


class AdaptiveInstanceNormalizationLoss(nn.Module):
    def __init__(self, style_feature, weight):
        super(AdaptiveInstanceNormalizationLoss, self).__init__()
        self.style_feature = style_feature
        self.weight = weight
        self.loss = F.mse_loss(self.style_feature, self.style_feature)

    def forward(self, content_feature):
        normalized_feature = adaptive_instance_normalization(content_feature, self.style_feature)
        self.loss = F.mse_loss(content_feature * self.weight, normalized_feature)
        return normalized_feature

def get_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img):
    normalization = Normalization(normalization_mean, normalization_std).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    model = nn.Sequential(normalization)
    content_losses = []
    style_losses = []

    style_feature = cnn(style_img).detach()
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target, weight=1)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            adain_loss = AdaptiveInstanceNormalizationLoss(target, weight=1e6)
            model.add_module(f'adain_loss_{i}', adain_loss)
            style_losses.append(adain_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], AdaptiveInstanceNormalizationLoss):
            break

    model = model[:i + 1]

    return model, style_losses, content_losses

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std