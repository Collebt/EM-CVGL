from torch.optim import lr_scheduler




class Scheduler:
    def __init__(self, optimizers:list, step_size=200, gamma=0.1) -> None:
        self.opts = optimizers
        self.sches = []
        for opt in optimizers:
            self.sches.append(lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma))


    def step(self):
        for sche in self.sches:
            sche.step()
            

    def get_last_lr(self):
        lr = self.sches[0].get_last_lr()
        return lr
