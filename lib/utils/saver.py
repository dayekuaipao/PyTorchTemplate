import os

import torch


class Saver(object):
    def __init__(self, args):
        self.args = args
        self.save_path = args.save_path

    def save_checkpoint(self, checkpoint, filename='checkpoint.pth'):
        filename = os.path.join(self.save_path, filename)
        torch.save(checkpoint, filename)

    def save_parameters(self):
        with open(os.path.join(self.save_path, 'parameters.txt'), 'w') as f:
            f.write(str(self.args))
