from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, end_epoch, last_epoch=-1):
        self.end_epoch = end_epoch
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr / self.end_epoch * (self.last_epoch+1)
                for base_lr in self.base_lrs]
class WarmUpStepLR(_LRScheduler):
    def __init__(self, optimizer, warm_up_end_epoch, step_size, gamma=0.1,last_epoch=-1):
        self.warm_up_end_epoch = warm_up_end_epoch
        self.step_size = step_size
        self.gamma = gamma
        super(WarmUpStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):       
        return [base_lr / self.warm_up_end_epoch * (self.last_epoch )
                if self.last_epoch<self.warm_up_end_epoch 
                else base_lr * self.gamma**((self.last_epoch-self.warm_up_end_epoch)//self.step_size) 
                for base_lr in self.base_lrs]

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch,power=0.9,last_iter=-1):
        self.num_epochs=num_epochs
        self.iters_per_epoch=iters_per_epoch
        self.power=power
        super(PolyLR,self).__init__(optimizer, last_iter)
    def get_lr(self):
        return [base_lr *(1-self.last_epoch/(self.iters_per_epoch*self.num_epochs))**self.power
        for base_lr in self.base_lrs]