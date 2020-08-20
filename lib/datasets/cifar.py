import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import pickle
from lib.build.registry import Registries


@Registries.dataset_registry.register
class CIFAR10(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(CIFAR10, self).__init__()
        data_list = []
        label_list = []
        self.transform = transform
        self.target_transform = target_transform
        if split == 'train':
            for i in range(1, 5):
                filename = os.path.join(root, 'data_batch_%d' % (i,))
                with open(filename, 'rb') as f:
                    datadict = pickle.load(f, encoding='latin1')
                data = datadict['data']
                label = datadict['labels']
                data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
                label = np.array(label)
                data_list.append(data)
                label_list.append(label)
            self.data = np.concatenate(data_list)
            self.label = np.concatenate(label_list)
        elif split == 'valid':
            filename = os.path.join(root, 'data_batch_5')
            with open(filename, 'rb') as f:
                datadict = pickle.load(f, encoding='latin1')
            data = datadict['data']
            label = datadict['labels']
            self.data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            self.label = np.array(label)
        elif split == 'test':
            filename = os.path.join(root, 'test_batch')
            with open(filename, 'rb') as f:
                datadict = pickle.load(f, encoding='latin1')
            data = datadict['data']
            label = datadict['labels']
            self.data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            self.label = np.array(label)
        else:
            raise RuntimeError('The split must be train,valid or test in this dataset,but it got {}. '.format(split))

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            data = self.transform(data)  # 在这里做transform，转为tensor等等
        if self.target_transform is not None:
            label = self.target_transform(label)
        sample = {'data': data, 'label': label}
        return sample

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as F

    path = '../data/cifar-10-batches-py/'
    batch_size = 2
    transform = transforms.Compose(
        [transforms.ToTensor(),
         ])
    train_dataset = CIFAR10(root=path, split='train', transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    dataiter = iter(train_loader)
    sample = dataiter.next()
    images, labels = sample['data'], sample['label']
    img_grid = torchvision.utils.make_grid(images.float())
    img_grid = F.to_pil_image(img_grid)
    img_grid.save("pic.jpg")
