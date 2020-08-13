import numpy as np
import torch
import torch.nn as nn
from PIL.ImageDraw import Draw
from PIL import Image



def draw_pose(pred_model, testLoader, n_images, device):
    img_name, img, target = next(iter(testLoader))
    pred_model.eval()
    img = img.to(device)
    outputs = pred_model(img)
    num_joints = 14
    images = []
    for i in range(n_images):
        true_points = target[i].cpu().detach().numpy().reshape(-1, num_joints)*199
        pred_points = outputs[i].cpu().detach().numpy().reshape(-1, num_joints)*199
        path = img_name[i]
        image = Image.open(path)
        draw = Draw(image)
        for i in range(2):
            for j in range(5):
                start_point_true = (true_points[0, 6*i+j], true_points[1, 6*i+j])
                end_point_true = (true_points[0, 6*i+j+1], true_points[1, 6*i+j+1])
                start_point_pred = (pred_points[0, 6*i+j], pred_points[1, 6*i+j])
                end_point_pred = (pred_points[0, 6*i+j+1], pred_points[1, 6*i+j+1])
                draw.line((start_point_true, end_point_true), fill=(0, 255, 0), width=3)
                draw.line((start_point_pred, end_point_pred), fill=(255, 0, 0), width=3)
        start_point_true = (true_points[0, 12], true_points[1, 12])
        end_point_true = (true_points[0, 13], true_points[1, 13])
        start_point_pred = (pred_points[0, 12], pred_points[1, 12])
        end_point_pred = (pred_points[0, 13], pred_points[1, 13])
        draw.line((start_point_true, end_point_true), fill=(0, 255, 0), width=3)
        draw.line((start_point_pred, end_point_pred), fill=(255, 0, 0), width=3)
        images.append(image)
    return images



def generate_maps(points, height, width, g_model, device):
    points_map = torch.zeros((points.size(0), points.size(-1), height, width)).to(device)
    points = points.transpose(1, 2).contiguous().int()
    for i in range(points.size(0)):
        for j in range(points.size(1)):
            points_map[i, j, points[i, j, 1], points[i, j, 0]] = 256
    points_map = g_model(points_map)
    return points_map

def match_format(dic):
    loc = dic['loc_k'][:,:,0,:]
    joint_2d = np.zeros((np.shape(loc)[0], 2, 14))
    for i in range(np.shape(loc)[0]):
        for j in range(14):
            joint_2d[i][0][j] = loc[i][j][0]
            joint_2d[i][1][j] = loc[i][j][1]

    return joint_2d


class Heatmaps_to_Joints:
    def __init__(self):
        self.pool = nn.MaxPool2d(21, 1, 10) #3,1,1


    def find_max(self, heatmap):
        maxm = self.pool(heatmap)
        maxm = torch.eq(maxm, heatmap).float()
        heatmap = heatmap * maxm
        return heatmap

    def calc(self, heatmap):
        with torch.no_grad():
            heatmap = torch.autograd.Variable(torch.Tensor(heatmap))

        heatmap = self.find_max(heatmap)
        w = heatmap.size()[3]
        heatmap = heatmap.view(heatmap.size()[0], heatmap.size()[1], -1)
        val_k, ind = heatmap.topk(1, dim=2)

        x = ind % w
        y = (ind // w).long()
        ind_k = torch.stack((x, y), dim=3)
        answer = {'loc_k': ind_k, 'val_k': val_k}

        return {key:answer[key].cpu().data.numpy() for key in answer}

    def search(self, heatmap):
        result = self.calc(heatmap)
        joint_2d = match_format(result)

        return joint_2d

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
