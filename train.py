import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader,sampler
from lib.build.registry import Registries
from lib.datasets.cifar import *
from lib.models.backbones.resnet import *
import tqdm
from torch.nn import functional as F
from lib.utils.saver import Saver
from lib.utils.logger import Logger
from lib.utils.evaluator import Evaluator
from lib.utils.lr_scheduler import WarmUpStepLR
from lib.utils import transforms
from lib.utils.loss import FocalLoss
class Trainer:
        def __init__(self,args):
                self.args=args         

                # Define Saver
                self.saver = Saver(args)
                
                # Define Logger
                self.logger = Logger(args.save_path)

                # Define Evaluator
                self.evaluator = Evaluator(args.num_classes)

                # Define Best Prediction
                self.best_pred = 0.0
                
                # Define Dataloader
                train_transform=transforms.Compose([
                transforms.ToTensor(),
                ])
                valid_transform=transforms.Compose([
                transforms.ToTensor(),
                ])
                dataset = Registries.dataset_registry.__getitem__(args.dataset)(args.dataset_path,'train',train_transform) 
                
                # Define number of training data
                self.num_train = args.num_train
                
                if self.num_train == -1:
                    self.num_train = dataset.__len__()
                    
                kwargs = {
                    'batch_size':args.batch_size,
                    'num_workers': args.num_workers, 
                    'pin_memory': True}
                
                self.train_loader = DataLoader(dataset=dataset, 
                                    shuffle=False ,
                                    sampler=sampler.SubsetRandomSampler(range(self.num_train)),**kwargs)
                self.valid_loader = DataLoader(dataset=dataset, 
                                     shuffle=False,
                                     sampler=sampler.SubsetRandomSampler(range(self.num_train,dataset.__len__())),**kwargs)

                # Define Model
                self.model = Registries.backbone_registry.__getitem__(args.backbone)(num_classes=10)

                # Define Optimizer
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.init_learning_rate, momentum=0.9, dampening=0.1)   
                
                # Define Criterion
                self.criterion = FocalLoss()     
               
                # Define  Learning Rate Scheduler
                self.scheduler = WarmUpStepLR(self.optimizer, warm_up_end_epoch=0,step_size=50, gamma=0.1)     
                
                # Use cuda
                if torch.cuda.is_available() and args.use_gpu:
                        self.device = torch.device("cuda",args.gpu_ids[0])
                        if len(args.gpu_ids) > 1:
                                self.model = torch.nn.DataParallel(self.model, device_ids=args.gpu_ids)     
                else:
                        self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                
                # Use pretrained model
                if args.pretrained_model_path is not None:
                        if not os.path.isfile(args.pretrained_model_path):
                                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.pretrained_model_path))
                        else:                      
                                checkpoint = torch.load(args.pretrained_model_path)
                                if args.use_gpu and len(args.gpu_ids) > 1:
                                        self.model.module.load_state_dict(checkpoint['model'])
                                else:
                                        self.model.load_state_dict(checkpoint['model'])
                                self.scheduler.load_state_dict(checkpoint['scheduler'])
                                self.best_pred = checkpoint['best_pred']
                                self.optimizer = self.scheduler.optimizer
                                epoch = checkpoint['epoch'],
                                print("=> loaded checkpoint '{}'".format(args.pretrained_model_path))

        def train(self,epoch):  
                print('train epoch %d' % epoch)
                total_loss=0                                           
                tbar = tqdm.tqdm(self.train_loader)
                self.model.train()   #change the model to train mode
                step_num = len(self.train_loader)
                for step,sample in enumerate(tbar):
                        inputs,labels = sample['data'],sample['label']      #get the inputs and labels from dataloader     
                        inputs,labels = inputs.to(self.device),labels.to(self.device)   
                        if epoch == 0 and step == 0:
                                self.logger.show_img_grid(inputs)
                                self.logger.writer.add_graph(self.model, inputs)                 
                        self.optimizer.zero_grad()   #zero the optimizer because the gradient will accumulate in PyTorch
                        outputs = self.model(inputs)   #get the output(forward)              
                        loss = self.criterion(outputs, labels)       #compute the loss
                        loss.backward() #back propagate the loss(backward)
                        total_loss+=loss.item()
                        self.optimizer.step()        #update the weights
                        tbar.set_description('train iteration loss= %.6f' % loss.item())
                        self.logger.writer.add_scalar('train iteration loss', loss, epoch*step_num+step)
                self.logger.writer.add_scalar('train epoch loss', total_loss/step_num, epoch)
                self.scheduler.step()        #update the learning rate
                self.saver.save_checkpoint({'scheduler':self.scheduler.state_dict(),
                                   'state_dict':self.model.state_dict(),
                                   'best_pred':self.best_pred,
                                   'epoch':epoch},
                                   'current_checkpoint.pth')
        def valid(self,epoch):
                print('valid epoch %d' % epoch)
                total_loss=0                                           
                tbar = tqdm.tqdm(self.valid_loader)
                self.model.eval()   #change the model to eval mode
                step_num = len(self.valid_loader)
                with torch.no_grad():
                        for step,sample in enumerate(tbar):
                                inputs,labels = sample['data'],sample['label']     #get the inputs and labels from dataloader     
                                inputs,labels = inputs.to(self.device),labels.to(self.device)                    
                                outputs = self.model(inputs)   #get the output(forward)              
                                loss = self.criterion(outputs, labels)       #compute the loss
                                total_loss+=loss.item()
                                predicts= torch.argmax(outputs,dim=1)
                                tbar.set_description('valid iteration loss= %.6f' % loss.item())
                                self.logger.writer.add_scalar('valid iteration loss', loss, epoch*step_num+step)
                                self.evaluator.add_batch(labels.cpu().numpy(),predicts.cpu().numpy())
                self.logger.writer.add_scalar('valid epoch loss', total_loss/step_num, epoch)
                new_pred=self.evaluator.Mean_Intersection_over_Union()
                
                if new_pred > self.best_pred:
                        self.best_pred = new_pred
                        self.saver.save_checkpoint({'scheduler': self.scheduler.state_dict(),
                                           'model':self.model.state_dict(),
                                           'best_pred':self.best_pred,
                                           'epoch':epoch},
                                           'best_checkpoint.pth')
                        self.saver.save_parameters()



def main():
        # basic parameters
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
        parser.add_argument('--start_epoch', type=int, default=0, help='Start counting epochs from this number')
        parser.add_argument('--valid_step', type=int, default=1, help='How often to perform validation (epochs)')
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset you are using.')
        parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone you are using.')
        parser.add_argument('--model', type=str, default='resnet18', help='Model you are using.')
        parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
        parser.add_argument('--init_learning_rate', type=float, default=0.001, help='init learning rate used for train')
        parser.add_argument('--dataset_path', type=str, default='./data/cifar-10-batches-py/',help='path to dataset')
        parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
        parser.add_argument('--num_classes', type=int, default=10, help='num of object classes (with void)')
        parser.add_argument('--num_train', type=int, default=-1, help='num of training data')
        parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ids used for training')
        parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
        parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to load pretrained model')
        parser.add_argument('--save_path', type=str, default=os.getcwd(), help='path to save pretrained model and results')
        parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
        args = parser.parse_args()
        if args.use_gpu:
                try:
                        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
                except ValueError:
                        raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

        # start to train
        print(args)
        torch.manual_seed(args.seed)
        trainer = Trainer(args)
        for epoch in range(trainer.args.start_epoch, trainer.args.num_epochs):
                trainer.train(epoch)
                if epoch % args.valid_step == (args.valid_step - 1) and args.num_train != -1:
                        trainer.valid(epoch)

if __name__ == '__main__':
    main()