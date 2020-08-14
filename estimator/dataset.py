from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
import matplotlib.image as mpimage
from PIL import Image, ImageFilter, ImageDraw
import imgaug.augmenters as iaa
import imageio
import os
import numpy as np
from scipy.io import loadmat
import torch

def augment(aug_prob, images_path, data_path,
            img_dist_path='./augmented_images', data_dist_path='./aumented_joints.pkl'):
    if os.path.isfile(data_dist_path) is True:
        return img_dist_path, data_dist_path
    with open(data_path, 'rb') as joints_file:
        joints = pickle.load(joints_file)
    seq = iaa.Sequential([
        iaa.SomeOf((1, 4), [
            iaa.Affine(scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},),
            iaa.Affine(rotate=(-45, 45),),
            iaa.Affine(shear=(-16, 16),),
            iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
            iaa.Add((-20, 20), per_channel=0.5),
        ]),
    ])
    image_names = sorted(os.listdir(images_path))
    img_counter = 0
    new_joints = []
    for img_file in image_names:
        # print(img_file)
        choice = np.random.choice([True, False], p=[aug_prob, 1-aug_prob])
        if choice:
            new_joint = []
            img = Image.open(os.path.join(images_path, img_file))
            img = np.array([np.array(img)])
            key_points = np.array([np.column_stack((joints[img_counter, 0, :], joints[img_counter, 1, :]))])
            aug_image, points_aug = seq(images=img, keypoints=key_points)
            points_aug = np.row_stack((points_aug[0, :, 0], points_aug[0, :, 1]))
            new_joint.append(points_aug[0])
            new_joint.append(points_aug[1])
            new_path = os.path.join(img_dist_path, 'aaa_'+img_file)
            imageio.imwrite(new_path, aug_image[0])
            new_joints.append(new_joint)
        img_counter += 1
    new_joints = np.array(new_joints)
    new_joints = np.vstack((new_joints, joints))
    joints_file = open(data_dist_path, 'wb')
    pickle.dump(new_joints, joints_file)
    joints_file.close()
    return img_dist_path, data_dist_path


def resize_dataset(resize_shape, images_path, data_path,
                   img_dist_path='./rescaled_images',
                   data_dist_path='./rescaled_joints.pkl'):
    if os.path.isfile(data_dist_path) is True:
        return img_dist_path, data_dist_path
    os.mkdir(img_dist_path)
    with open(data_path, 'rb') as joints_file:
        joints = pickle.load(joints_file)
    new_joints = np.zeros(joints.shape)
    image_names = sorted(os.listdir(images_path))
    img_counter = 0
    for img_file in image_names:
        img = Image.open(os.path.join(images_path, img_file))
        img_shape = img.size
        scale_x = resize_shape[0] / img_shape[0]
        scale_y = resize_shape[1] / img_shape[1]
        img = img.resize(resize_shape, Image.BICUBIC)
        new_joints[img_counter, 0, :] = joints[img_counter, 0, :] * scale_x
        new_joints[img_counter, 1, :] = joints[img_counter, 1, :] * scale_y
        new_path = os.path.join(img_dist_path, img_file)
        img.save(new_path, format='JPEG')
        img_counter += 1
    joints_file = open(data_dist_path, 'wb')
    new_joints[new_joints>(resize_shape[0]-1)] = resize_shape[0]-1
    pickle.dump(new_joints, joints_file)
    joints_file.close()
    return img_dist_path, data_dist_path


def transform_joints(images_folder, joints_path, aug_prob=None, resize_shape=None):
    joints = loadmat(joints_path)['joints'][:2].transpose(2, 0, 1)
    joints_path = './joints.pkl'
    joints_file = open(joints_path, 'wb')
    pickle.dump(joints, joints_file)
    joints_file.close()
    if aug_prob:
        joints_dist = './augmented_joints.pkl'
        images_folder, joints_path = augment(aug_prob, images_folder, joints_path, images_folder, joints_dist)
    if resize_shape:
        joints_dist = './rescaled_joints.pkl'
        images_dist = './rescaled_images'
        images_folder, joints_path = resize_dataset(resize_shape, images_folder, joints_path, images_dist, joints_dist)
    return images_folder, joints_path


class LSPDataset(Dataset):

    def __init__(self, images_path, data_path, transform=ToTensor(), resize_shape=None, aug_prob=None):
        if aug_prob:
            images_path, data_path = augment(aug_prob, images_path, data_path)
        if resize_shape:
            images_path, data_path = resize_dataset(resize_shape, images_path, data_path)
        self.image_names = [
            os.path.join(images_path, img_name) for img_name in sorted(os.listdir(images_path))
        ]
        with open(data_path, 'rb') as joints_file:
            self.joints = pickle.load(joints_file)
        self.transform = transform

    def __getitem__(self, item):
        img_name = self.image_names[item]
        img = Image.open(img_name)
        if self.transform:
            img = self.transform(img)

        joints_pos = torch.as_tensor(self.joints[item], dtype=torch.float)
        return img_name, img, joints_pos

    def __len__(self):
        return len(self.image_names)


def get_data_loaders(data_path, splits, batch_sizes, shuffle_images=True, seed=None, transform=None, **kwargs):
    main_dataset = LSPDataset(data_path[0], data_path[1], transform=transform)
    n_samples = len(main_dataset)
    indices = list(range(n_samples))
    train_index = int(np.floor(splits[0]*n_samples))
    valid_index = int(np.floor((splits[0]+splits[1])*n_samples))
    if shuffle_images:
        if seed:
            np.random.seed(seed)
        np.random.shuffle(indices)
    tr_indices = indices[:train_index]
    val_indices = indices[train_index:valid_index]
    ts_indices = indices[valid_index:]
    tr_sampler = SubsetRandomSampler(tr_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    ts_sampler = SubsetRandomSampler(ts_indices)
    train_loader = DataLoader(main_dataset, batch_size=batch_sizes[0], shuffle=False, sampler=tr_sampler, **kwargs)
    valid_loader = DataLoader(main_dataset, batch_size=batch_sizes[1], shuffle=False, sampler=val_sampler, **kwargs)
    test_loader = DataLoader(main_dataset, batch_size=batch_sizes[2], shuffle=False, sampler=ts_sampler, **kwargs)
    return train_loader, valid_loader, test_loader
