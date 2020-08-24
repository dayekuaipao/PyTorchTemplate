import os

import torchvision
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, path):
        self.writer = SummaryWriter(os.path.join(path, 'runs'))

    def show_img_grid(self, images, info='pictures'):
        """
        show a batch of images
        images:a batch of tensors which are a batch of images
        info:the information about the pictures
        """
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image(info, img_grid)

    def add_projector(self, feature, label=None, img=None):
        """
        visualize the lower dimensional representation of higher dimensional data
        feature:a matrix which each row is the feature vector of the data point
        label:label
        img:label_img
        """
        self.writer.add_embedding(mat=feature, metadata=label, label_img=img)

    def draw_scalars(self, tag, scalars, start_iteration=0):
        """
         draw the list curve,such as loss accuracy
         list:the list of values
         start_iteration:the start iteration of training,default is 0,means train from scratch
        """
        for i, item in enumerate(scalars):
            self.writer.add_scalar(tag, item, start_iteration + i)

    def add_pr_curve_tensorboard(self, tag, labels, predictions, global_step=0):
        """
        Takes in a class index and plots the corresponding
        precision-recall curve
        """
        self.writer.add_pr_curve(tag, labels, predictions, global_step=global_step)

    def close(self):
        self.writer.close()
