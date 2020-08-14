from estimator.models import Complete_Model, GaussianFilterLayer
from estimator.utils import Heatmaps_to_Joints, count_parameters, draw_pose
from estimator.train_test import train
from estimator.dataset import augment, transform_joints, resize_dataset, get_data_loaders
import argparse
from torchvision import transforms
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

if __name__ == '__main__':
    joint_name = ['Ankle:   ',
                  'Knee:    ',
                  'Hip:     ',
                  'Wrist:   ',
                  'Elbow:   ',
                  'Shoulder:',
                  'Head:     ']

    parser = argparse.ArgumentParser(description='Pytorch Fast Human Pose Estimation!')
    parser.add_argument('--train_batch_size', type=int, default=12,
    help='train batch size (default: 12)')
    parser.add_argument('--num_joints', type=int, default=14,
    help='number of joints (default: 14)')
    parser.add_argument('--epochs', type=int, default=10,
    help='number of epochs (default: 10)')
    parser.add_argument('--num_stacks', type=int, default=4,
    help='number of stacked networks (default: 4)')
    parser.add_argument('--seed', type=int, default=12568,
    help='seed (default: 12568)')
    parser.add_argument('--num_network_channels', type=int, default=64,
    help='number of network channels (default: 64)')
    parser.add_argument('--images_path', type=str, default='lsp_dataset/images/',
    help='images path')
    parser.add_argument('--joints_path', type=str, default='lsp_dataset/joints.mat',
    help='joints path')
    parser.add_argument('--resize_shape', type=int, default=256,
    help='resize shape (default: 256)')
    parser.add_argument('--aug_prob', type=float, default=0.2,
    help='augmentation probability (default: 0.2)')
    parser.add_argument('--device', type=bool, default=True,
    help='device (default: True)')
    parser.add_argument('--train_percentage', type=float, default=0.7,
    help='train percentage (default: 0.7)')
    parser.add_argument('--valid_percentage', type=float, default=0.1,
    help='validation percentage (default: 0.1)')
    parser.add_argument('--test_percentage', type=float, default=0.2,
    help='test percentage (default: 0.2)')
    parser.add_argument('--shuffle', type=bool, default=False,
    help='shuffle (default: False)')
    parser.add_argument('--gussian_filter_size', type=int, default=21,
    help='guassain filter size (default: 21)')
    parser.add_argument('--gussian_sigma', type=float, default=3,
    help='gussian filter sigma (default: 3)')
    parser.add_argument('--lr', type=float, default=1e-4,
    help='learning rate (default: 1e-4)')
    parser.add_argument('--step_size', type=int, default=1,
    help='step size (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.9,
    help='scheduler gamma (default: 0.9)')

    args = parser.parse_args()

    images_folder = args.images_path
    joints_path = args.joints_path
    resize_shape = (args.resize_shape, args.resize_shape)
    aug_prob = args.aug_prob
    images_folder, joints_path = transform_joints(images_folder, joints_path, aug_prob=aug_prob, resize_shape=resize_shape)

    custom_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    cuda_flag = args.device
    seed = args.seed
    device = torch.device('cuda' if cuda_flag else 'cpu')
    if seed:
        torch.manual_seed(seed)
    params = {'num_workers': 0, 'pin_memory': True} if cuda_flag else {}
    data_paths = [images_folder, joints_path]
    data_splits = [args.train_percentage, args.valid_percentage, args.test_percentage]
    batch_sizes = [args.train_batch_size, args.train_batch_size, args.train_batch_size]
    train_loader, valid_loader, test_loader = get_data_loaders(data_paths, data_splits, batch_sizes, shuffle_images=args.shuffle,
                                                              seed=56238, transform=custom_trans, **params)


    model = Complete_Model(args.num_network_channels, args.train_batch_size, args.num_stacks, args.num_joints).to(device)
    g_model = GaussianFilterLayer(args.num_joints, args.gussian_filter_size, args.gussian_sigma, args.num_joints).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    joint_detection = Heatmaps_to_Joints()

    train_loss, valid_loss, PCKh_valid_accuracy = train(model, g_model, criterion, optimizer, train_loader, valid_loader,
                                                        args.epochs, device, scheduler, args.train_batch_size, joint_detection, joint_name)


    ######################################################################################
    plt.figure(figsize=(10,6), dpi=100)
    plt.title('Traning and Validation Loss')
    plt.plot(train_loss, color='blue', label='Train Loss')
    plt.plot(valid_loss, color='orange', label='Valid Loss')
    plt.legend()
    plt.savefig('/content/drive/My Drive/Colab Projects/Final Projects/Loss_Plot.jpg', dpi=200)
    plt.show()

    ######################################################################################
    #images = draw_pose(model, test_loader, 4)
