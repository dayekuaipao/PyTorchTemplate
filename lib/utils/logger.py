import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
import os
import torchvision


class Logger(object):
    def __init__(self, path):
        self.writer = SummaryWriter(os.path.join(path, 'runs'))

    def show_img_grid(self, images, info='pictures'):
        '''
        show a batch of images
        images:a batch of tensors which are a batch of images
        info:the information about the pictures
        '''
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image(info, img_grid)

    def add_projector(self, feature, label=None, img=None):
        '''
        visualize the lower dimensional representation of higher dimensional data
        mat:a matrix which each row is the feature vector of the data point 
        metadata:label
        label_img:img
        '''
        self.writer.add_embedding(mat=feature, metadata=label, label_img=img)

    def draw_list(self, list, start_iteration=0):
        '''
         draw the list curve,such as loss accuracy
         loss:the list of values
         start_iteration:the start iteration of training,default is 0,means train from scratch
        '''
        for i, item in enumerate(list):
            self.writer.add_scalar('training loss', item, start_iteration + i)

    @staticmethod
    def plot_classes_preds(output, labels):
        '''
        Generates matplotlib Figure using outputs and labels from a batch, 
        that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "output_to_probs" function.
        '''
        preds, probs = torch.max(output, dim=1)
        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(12, 48))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
            matplotlib_imshow(images[idx], one_channel=True)
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]],
                probs[idx] * 100.0,
                classes[labels[idx]]),
                color=("green" if preds[idx] == labels[idx].item() else "red"))
        return fig

    def add_pr_curve_tensorboard(class_index, labels, predictions, global_step=0):
        '''
        Takes in a class index and plots the corresponding
        precision-recall curve
        '''
        self.writer.add_pr_curve(class_index, labels, predictions, global_step=global_step)

    def close():
        self.writer.close()

    @staticmethod
    def pic_save(img, path='pic.jpg'):
        '''
        save torch tensor or numpy array as PIL picture
        '''
        img = to_pil_image(img)
        img.save(path)

    @staticmethod
    def pic_show(img):
        '''
        show torch tensor or numpy array as PIL picture
        '''
        img = to_pil_image(img)
        img.show()
