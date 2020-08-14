# Fast Human Pose Estimation
This project is an implementation of [Toward fast and accurate human pose estimation via soft-gated skip connections](https://arxiv.org/abs/2002.11098) using Pytorch

![](images/model_architecture.png)

The original paper first pre-trains the model by [MPII dataset](http://human-pose.mpi-inf.mpg.de/). Then it utilizes [LSP dataset](https://sam.johnson.io/research/lsp.html) to improve the model performance. Here we just use LSP dataset, however we augmente training data by various transformations, then all the images should resize to same dimensions before feeding to the model.
To train the model please download the dataset and extract it to specific location, finally pass both the images folder and joints.mat file as arguments to **main.py**. As an example consider the following command:
```
python main.py --images_path ./lsp_dataset/images/ --joints_path ./lsp_dataset/joints.mat
```
You would set other training options by passing more arguments. To check other options use the command below:
```
python main.py --help
```
Also we have created a notebook file named **main.ipynb** which would do the same task.
