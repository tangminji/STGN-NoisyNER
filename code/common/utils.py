import torch
import torch.nn.functional as F
import numpy as np
from tensorboard_logger import configure, log_value, log_histogram
import tensorboard_logger
import random
from tqdm import tqdm
import os
import shutil
import time
from cmd_args_noisyner import args
from torch.nn.utils import clip_grad_norm_

class AverageMeter_pnorm(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, pnorm):
        self.val = val
        self.sum += pow(val, pnorm)
        self.count += 1
        self.avg = pow(self.sum / self.count, 1./pnorm)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def compute_topk_accuracy(prediction, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        prediction (torch.Tensor): N*C tensor, contains logits for N samples over C classes.
        target (torch.Tensor):  labels for each row in prediction.
        topk (tuple of int): different values of k for which top-k accuracy should be computed.

    Returns:
        result (tuple of float): accuracy at different top-k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = prediction.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        result = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result

def generate_log_dir(args):
    """Generate directory to save artifacts and tensorboard log files."""

    print('\nLog is going to be saved in: {}'.format(args.log_dir))

    if os.path.exists(args.log_dir):
        if args.restart:
            print('Deleting old log found in: {}'.format(args.log_dir))
            shutil.rmtree(args.log_dir)
            configure(args.log_dir, flush_secs=10)
        else:
            error='Old log found; pass --restart flag to erase'.format(args.log_dir)
            raise Exception(error)
    else:
        configure(args.log_dir, flush_secs=10)

def generate_log_dir_hyp(args, ITERATION):
    """Generate directory to save artifacts and tensorboard log files."""
    '''for hyperopt iteration for MAX_EVALS times'''
    log_pth = os.path.join(args.log_dir, str(ITERATION))
    print('\nLog is going to be saved in: {}\n'.format(log_pth))

    if os.path.exists(log_pth):
        if args.restart:
            print('Deleting old log found in: {}'.format(log_pth))
            shutil.rmtree(log_pth)
            tensorboard_logger.clean_default_logger()
            configure(log_pth, flush_secs=10)
        else:
            error='Old log found; pass --restart flag to erase'.format(log_pth)
            raise Exception(error)
    else:
        #https://blog.csdn.net/webmater2320/article/details/105831920
        tensorboard_logger.clean_default_logger()
        configure(log_pth, flush_secs=10)

def set_seed(args):
    """Set seed to ensure deterministic runs.

    Note: Setting torch to be deterministic can lead to slow down in training.
    """
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_intermediate_iteration_stats(epoch, global_iter,
                                    losses, top1=None, top5=None):
    """
    dicrimloss for classification
    Log stats for data parameters and loss on tensorboard.
    """
    if top5 is not None:
        log_value('train_iteration_stats/accuracy_top5', top5.avg, step=global_iter)
    if top1 is not None:
        log_value('train_iteration_stats/accuracy_top1', top1.avg, step=global_iter)
    log_value('train_iteration_stats/loss', losses.avg, step=global_iter)
    log_value('train_iteration_stats/epoch', epoch, step=global_iter)

def log_stats(data, name, step):
    """Logs statistics on tensorboard for data tensor.

    Args:
        data (torch.Tensor): torch tensor.
        name (str): name under which stats for the tensor should be logged.
        step (int): step used for logging
    """
    log_value('{}/highest'.format(name), torch.max(data).item(), step=step)
    log_value('{}/lowest'.format(name), torch.min(data).item(),  step=step)
    log_value('{}/mean'.format(name), torch.mean(data).item(),   step=step)
    log_value('{}/std'.format(name), torch.std(data).item(),     step=step)
    try:
        log_histogram('{}'.format(name), data.data.cpu().numpy(),    step=step)
    except:
        print('xxx')

def log(path, str):
    print(str)
    with open(path, 'a') as file:
        file.write(str)

def log_hyp(path, cont, ITERATION):
    #{}/log.txt
    print(cont)
    tmp = path.split('/')
    path, logf = '/'.join(tmp[:-1]), tmp[-1]
    path = os.path.join(path, str(ITERATION), logf)
    with open(path, 'a+') as file:
        file.write(cont)

def checkpoint(acc, epoch, net, save_dir):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    file_path = save_dir + '/net.pth'
    torch.save(obj=state, f=file_path)

def hook_fn_moutput(grad):
    '''
    args.mode=GN_on_moutput
    '''
    if args.sigma > 0:
        #return grad + torch.div(args.sigma * (torch.rand(grad.size()) - 0.5).to(args.device), args.sm) if grad != None else grad
        #return grad + torch.div(args.sigma * torch.randn(grad.size()).to(args.device), args.sm) if grad != None else grad
        return grad + args.sigma * torch.randn(grad.size()).to(args.device) if grad != None else grad
    else:
        return grad

# # 记录信息
# def hook_fn_random_walk(grad):
#     if args.sigma > 0:
#         args.sigma_dyn[args.index] = torch.add(args.sigma_dyn[args.index].data, args.sign_loss+args.sign_forgetting_events,
#                                                alpha=args.lr_sig).detach()
#         # Clamp perturb variance within certain bounds
#         if args.skip_clamp_param is False:
#             # Project the sigma's to be within certain range
#             args.sigma_dyn.data = args.sigma_dyn.data.clamp_(
#                 min=0,
#                 max=args.sig_max)
#         # 必要时可以追踪扰动
#         perturb = torch.div(args.sigma_dyn[args.index].reshape((-1, 1)) * (torch.rand(grad.size()) - 0.5).to(
#             args.device), args.sm)
#         perturb.data = perturb.data.clamp_(
#             min=args.times *grad.min(),
#             max=args.times *grad.max()
#         )
#         return grad + perturb if grad != None else grad
#     else:
#         return grad

def hook_fn_random_walk(grad):
    if args.sigma > 0:
        args.sigma_dyn[args.index] = torch.add(args.sigma_dyn[args.index].data, args.sign_loss+args.sign_forgetting_events,
                                               alpha=args.lr_sig).detach()
        # Clamp perturb variance within certain bounds
        if args.skip_clamp_param is False:
            # Project the sigma's to be within certain range
            args.sigma_dyn.data = args.sigma_dyn.data.clamp_(
                min=0,
                max=args.sig_max)
        
        # perturb = torch.div(args.sigma_dyn[args.index].reshape((-1, 1)) * (torch.rand(grad.size()) - 0.5).to(
        #     args.device), args.sm)

        # 这里去掉sm分母试一下
        perturb = args.sigma_dyn[args.index].reshape((-1, 1)) * (torch.rand(grad.size()) - 0.5).to(
            args.device)

        perturb.data = perturb.data.clamp_(
            min=args.times *grad.min(),
            max=args.times *grad.max()
        )
        return grad + perturb if grad != None else grad
    else:
        return grad

def hook_fn_noisy_samples(grad):
    if args.sigma > 0:
        return grad + args.sigma * torch.tensor(args.obj).reshape((-1,1)).to(args.device)* torch.randn(grad.size()).to(args.device) if grad != None else grad
    else:
        return grad

def hook_fn_gods_perspective2(grad):
    if args.sigma > 0:
        args.sigma_dyn[args.index] = torch.add(args.sigma_dyn[args.index].data*torch.tensor(args.noisy).to(args.device),
                                               args.lr_sig)*torch.tensor(args.noisy).to(args.device).detach()+\
        torch.add(args.sigma_dyn[args.index].data*torch.tensor(args.clean).to(args.device),
                                               -args.lr_sig)*torch.tensor(args.clean).to(args.device).detach()
        # Clamp perturb variance within certain bounds
        if args.skip_clamp_param is False:
            # Project the sigma's to be within certain range
            args.sigma_dyn.data = args.sigma_dyn.data.clamp_(
                min=0,
                max=args.sig_max)
        return grad + args.sigma_dyn[args.index].reshape((-1, 1)) * (torch.rand(grad.size())-0.5).to(
            args.device) if grad != None else grad
    else:
        return grad

def hook_fn_parameter(grad):
    '''
    :param grad: z's gradient, z.register_hook(hook_fn)
    hook_fn(grad) -> Tensor or None
    add gaussian noise on leaf nodes
    '''
    #print(grad.size())
    if args.sigma > 0:
        #return a tuple with one element
        return grad + args.sigma * torch.randn(grad.size()).to(args.device) if grad != None else grad
    else:
        return grad

def adjust_learning_rate(model_initial_lr, optimizer, gamma, step):
    """Sets the learning rate to the initial learning rate decayed by 10 every few epochs.

    Args:
        model_initial_lr (int) : initial learning rate for model parameters
        optimizer (class derived under torch.optim): torch optimizer.
        gamma (float): fraction by which we are going to decay the learning rate of model parameters
        step (int) : number of steps in staircase learning rate decay schedule
    """
    lr = model_initial_lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

""" Training/testing """
# training
def train_noise(args, model, device, loader, optimizer):
    '''
    Train model for one epoch on the train set.
    '''
    model.train()
    train_loss = 0
    correct = 0
     
    for data, target in loader:
        
        if len(target.size())==1:
            target = torch.zeros(target.size(0), args.num_class).scatter_(1, target.view(-1,1), 1) # convert label to one-hot

        data, target = data.to(device), target.to(device)
            
        # SLN
        if args.sigma>0:
            target += args.sigma*torch.randn(target.size()).to(device)
        
        output = model(data)
        loss = -torch.mean(torch.sum(F.log_softmax(output, dim=1)*target, dim=1))
        
        optimizer.zero_grad()#set gradient to 0
        loss.backward()#compute gradient
        optimizer.step()

        train_loss += data.size(0)*loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        if len(target.size())==2: # soft target
            target = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    return train_loss/len(loader.dataset), correct/len(loader.dataset)

def GN_label_train(args, model, loader, criterion, optimizer, global_iter, epoch, logpath):
    '''
    SLN: Gaussian noise on the label noise
    train_for_one_epoch
    '''
    model.train()
    train_loss = AverageMeter('Loss', ':.4e')
    correct = AverageMeter('Acc@1', ':6.2f')#for classification
    t0 = time.time()
    for i, (data, target) in enumerate(tqdm(loader, unit='batch')):

        if len(target.size()) == 1:
            target = torch.zeros(target.size(0), args.num_class).scatter_(1, target.view(-1, 1),
                                                                          1)  # convert label to one-hot

        global_iter += 1
        data, target = data.to(args.device), target.to(args.device)

        #TODO: SLN
        if args.sigma > 0:
            target += args.sigma * torch.randn(target.size()).to(args.device)

        output = model(data)
        loss = -torch.mean(torch.sum(F.log_softmax(output, dim=1) * target, dim=1))
        #loss = criterion(output, target)

        optimizer.zero_grad()  # set gradient to 0
        loss.backward()  # compute gradient
        optimizer.step()

        # Measure accuracy and record loss
        train_loss.update(loss.item(), data.size(0))
        if len(target.size()) == 2:  # soft target
            target = target.argmax(dim=1, keepdim=True)
        #pred = output.argmax(dim=1, keepdim=True)
        #acc1 = pred.eq(target.view_as(pred)).float().sum().mul_(100.0 / data.size(0))
        acc1 = compute_topk_accuracy(output, target, topk=(1,))
        correct.update(acc1[0].item(), data.size(0))

        # Log stats for data parameters and loss every few iterations
        if i % args.print_freq == 0:
            log_intermediate_iteration_stats(epoch, global_iter, train_loss, top1=correct)

    # Print and log stats for the epoch
    log_value('train/loss', train_loss.avg, step=epoch)
    log(logpath, 'Time for Train-Epoch-{}/{}:{:.1f}s Acc:{}, Loss:{}\n'.format(epoch, args.epochs, time.time() - t0,
                                                                             correct.avg, train_loss.avg))
    log_value('train/accuracy', correct.avg, step=epoch)
    return global_iter, train_loss.avg, correct.avg

def GN_model_train(args, model, loader, criterion, optimizer, global_iter, epoch, logpath):
    '''
    Gaussian noise on the gradient of loss w.r.t the model architecture(on layers)
    Gaussian noise on the gradient of loss w.r.t parameters
    Gaussian noise on the gradient of loss w.r.t the model output
    train_for_one_epoch
    '''
    model.train()
    train_loss = AverageMeter('Loss', ':.4e')
    correct = AverageMeter('Acc@1', ':6.2f')#for classification
    t0 = time.time()
    for i, (data, target) in enumerate(tqdm(loader, unit='batch')):
        global_iter += 1
        data, target = data.to(args.device), target.to(args.device)

        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()

        # Measure accuracy and record loss
        train_loss.update(loss.item(), data.size(0))
        acc1 = compute_topk_accuracy(output, target, topk=(1,))
        correct.update(acc1[0].item(), data.size(0))

        # Log stats for data parameters and loss every few iterations
        if i % args.print_freq == 0:
            log_intermediate_iteration_stats(epoch, global_iter, train_loss, top1=correct)

    # Print and log stats for the epoch
    log_value('train/loss', train_loss.avg, step=epoch)
    log(logpath, 'Time for Train-Epoch-{}/{}:{:.1f}s Acc:{}, Loss:{}\n'.format(epoch, args.epochs, time.time() - t0, correct.avg, train_loss.avg))
    log_value('train/accuracy', correct.avg, step=epoch)
    return global_iter, train_loss.avg, correct.avg

def validate(args, model, loader, test_best, criterion, epoch, logpath, mode='val'):
    '''
    Evaluates model on validation/test set and logs score on tensorboard.
    '''
    test_loss = AverageMeter('Loss', ':.4e')
    correct = AverageMeter('Acc@1', ':6.2f')#for classification
    # switch to evaluate mode
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(args.device), target.to(args.device)

            # compute output
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            test_loss.update(loss.item(), data.size(0))
            acc1 = compute_topk_accuracy(output, target, topk=(1,))
            correct.update(acc1[0].item(), data.size(0))
    log(logpath, 'Time for {}-Epoch-{}/{}:{:.1f}s Acc:{}, Loss:{}\n'.format('Test'if mode=='test'else 'Val',
                                                                      epoch, args.epochs, time.time()-t0, correct.avg, test_loss.avg))
    log_value('{}/loss'.format(mode), test_loss.avg, step=epoch)
    # Logging results on tensorboard
    log_value('{}/accuracy'.format(mode), correct.avg, step=epoch)
    # Save checkpoint.
    acc = correct.avg
    if acc > test_best:
        test_best = acc
        checkpoint(acc, epoch, model, args.log_dir)

    return test_best, test_loss.avg, correct.avg

# testing
def test(args, model, device, loader, top5=False, criterion=F.cross_entropy):
    model.eval()
    test_loss = 0
    correct = 0
    correct_k = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            if top5:
                _, pred = output.topk(5, 1, True, True)
                correct_k += pred.eq(target.view(-1,1)).sum().item()
    if top5:
        return test_loss/len(loader.dataset), correct_k/len(loader.dataset)
    else:
        return test_loss/len(loader.dataset), correct/len(loader.dataset)
 

def get_output(model, device, loader):
    softmax_outputs = []
    losses = []
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if len(target.size())==1:
                loss = F.cross_entropy(output, target, reduction='none')
            else:
                loss = -torch.sum(F.log_softmax(output, dim=1)*target, dim=1)
            output = F.softmax(output, dim=1)
               
            losses.append(loss.cpu().numpy())
            softmax_outputs.append(output.cpu().numpy())
            
    return np.concatenate(softmax_outputs), np.concatenate(losses)



# get hidden features (before the final fully-connected layer)
def get_feat(model, device, loader):
    feats = []
    model.eval()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            feat = model(data, get_feat=True)
            feats.append(feat.cpu().numpy())
    return np.concatenate(feats)


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        #self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):                   
            # fix the error 'RuntimeError: result type Float can't be cast to the desired output type Long'
            #print(param.type())
            if param.type()=='torch.cuda.LongTensor':
                ema_param = param
            else:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            #param.mul_(1 - self.wd)