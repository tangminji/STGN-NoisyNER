import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# #os.environ["CUDA_VISIBLE_DEVICES"] = "0"#"0,1,2"

import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from common.utils import log, set_seed, generate_log_dir, AverageMeter, \
    compute_topk_accuracy, checkpoint, log_intermediate_iteration_stats, log_stats
from common.utils import hook_fn_random_walk
from cmd_args_noisyner import args,TRAIN_SETTINGS,EXP_SETTINGS
from tensorboard_logger import log_value
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data.NoisyNER_dataset import get_noisyner_dataset, get_NoisyNER_model_and_loss_criterion
import json
from hyperopt import STATUS_OK
from ner_datacode import LabelRepresentation, Evaluation
import logging
import csv

first = True
counter = 0
sig_max = args.sig_max
delta = 0#判断是否满足增速

MD_CLASSES = {
    'NoisyNER':(get_noisyner_dataset, get_NoisyNER_model_and_loss_criterion)
}

# {"O": 0, "PER": 1, "ORG": 2, "LOC": 3}
# {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "B-MISC": 4, "I-ORG": 5, "B-LOC": 6, "I-LOC": 7, "I-MISC": 8}

bio_to_io = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 2,
    6: 3,
    7: 3,
    8: 4
}

def convert_bio_to_io_idx(labels):
    io_labels = torch.tensor([bio_to_io[label] for label in labels.tolist()])
    return io_labels

def train_others(args, model, loader, optimizer, criterion, global_iter, epoch, logpath):
    '''
    Gaussian noise on the gradient of loss w.r.t parameters
    Gaussian noise on the gradient of loss w.r.t the model output
    train_for_one_epoch
    '''
    model.train()
    train_loss = AverageMeter('Loss', ':.4e')
    correct = AverageMeter('Acc@1', ':6.2f')#for classification
    fcorrect = AverageMeter('Acc@1', ':6.2f')
    tcorrect = AverageMeter('Acc@1', ':6.2f')
    t0 = time.time()
    #loss_lst = TDigest()
    loss_lst = []
    for i, (data, target, word, target_gt, index) in enumerate(loader):
        
        global_iter += 1
        # similar to global variable
        args.index = index
        # if len(target.size()) == 1:
        #     target = torch.zeros(target.size(0), args.num_class).scatter_(1, target.view(-1, 1),
        #                                                                   1)  # convert label to one-hot
        data, target = data.to(args.device), target.to(args.device)

        output = model(data)
        args.sm = F.softmax(output)
        # SLN
        if args.mode == 'GN_on_label':
            onehot = F.one_hot(target.long(), args.num_class).float()
            onehot += args.sigma*torch.randn(onehot.size()).to(args.device) 
            loss = -torch.sum(F.log_softmax(output, dim=1)*onehot, dim=1)
        else:
            if args.mode == 'Random_walk':
                output.register_hook(hook_fn_random_walk)

            loss = criterion(output, target)
            if args.mode == 'Random_walk':
                # TODO: element1: from loss perspective
                # TODO: quantile
                loss_lst.append(loss.detach().cpu().numpy().tolist())
                if len(loss_lst) > args.avg_steps:
                    loss_lst.pop(0)
                losses = sum(loss_lst,[])
                k1 = torch.quantile(torch.tensor(losses).to(args.device),
                                    1 - args.drop_rate_schedule[args.cur_epoch - 1])
                #TODO: element2: from forgetting events perspective, see algorithm 1 in ICLR19 an empirical study of example...
                _, predicted = torch.max(output.data, 1)
                # Update statistics and loss
                acc = (predicted == target).to(torch.long)
                forget_or_not = torch.gt(args.prev_acc[index], acc)#greater than
                args.forgetting[index] = args.forgetting[index] + forget_or_not
                args.prev_acc[index] = acc

                #when to update, since forgetting times of any sample reaches to args.forget_times
                # if args.forget_times in args.forgetting:
                times_ge_or_not = torch.ge(args.forgetting[index], args.forget_times).detach()
                if times_ge_or_not.any():
                    #greater or equal
                    args.sign_forgetting_events = ((1-args.ratio_l)*args.total) * torch.tensor([1 if t == True else -1 for t in times_ge_or_not]).to(args.device)
                    args.sign_loss = (args.ratio_l * args.total) * torch.sign(loss - k1).to(args.device)
                else:
                    args.sign_forgetting_events = torch.tensor([0]*len(loss)).to(args.device)
                    if args.ratio_l != 0:
                        args.sign_loss = torch.sign(loss - k1).to(args.device)
                    else:
                        args.sign_loss = torch.tensor([0] * len(loss)).to(args.device)

        loss = loss.mean()
        #new_l = new_l.mean()
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()

        args.sm = None # 这里清空一下，避免爆空间
        # 这里评价记录了ACC
        # 由于是碎片化比较，没有连续文章，所以无法记录F1
        # Measure accuracy and record loss
        train_loss.update(loss.item(), data.size(0))
        
        pred = output.argmax(dim=1) # 这里不要keepdim
        # noise & disturbance ground-truth index

        target_gt = target_gt.to(args.device)
        gt = target==target_gt
        agree = pred==target
        fc = agree[~gt]
        tc = agree[gt]

        if(len(fc)>0):
            fcorrect.update(fc.sum().item() * (100.0 / len(fc)), len(fc))
        if(len(tc)>0):
            tcorrect.update(tc.sum().item() * (100.0 / len(tc)), len(tc))

        acc1 = compute_topk_accuracy(output, target, topk=(1,))
        correct.update(acc1[0].item(), data.size(0))

        # Log stats for data parameters and loss every few iterations
        if i % args.print_freq == 0:
            log_intermediate_iteration_stats(epoch, global_iter, train_loss, top1=correct)

    # Print and log stats for the epoch
    log_value('train/loss', train_loss.avg, step=epoch)
    log(logpath, 'Time for Train-Epoch-{}/{}:{:.1f}s Acc:{}, Loss:{}\n'.
            format(epoch, args.epochs, time.time() - t0, correct.avg, train_loss.avg))

    log_value('train/accuracy', correct.avg, step=epoch)
    log_value('train/true_correct_from_clean', tcorrect.avg, step=epoch)
    log_value('train/false_correct_from_noise_disturb', fcorrect.avg, step=epoch)
    return global_iter, train_loss.avg, correct.avg, tcorrect.avg, fcorrect.avg

# 注意：评价时要评价F1值
# 这里算loss和acc时都用io标签，和训练时保持一致
def validate(args, model, loader, criterion, epoch, logpath, raw_data, label_representation, mode='val'):
    '''
    Evaluates model on validation/test set and logs score on tensorboard.
    '''
    test_loss = AverageMeter('Loss', ':.4e')
    correct = AverageMeter('Acc@1', ':6.2f')#for classification
    # switch to evaluate mode
    model.eval()
    t0 = time.time()
    predictions = []
    words = []

    test_loss.update(0,1)
    correct.update(0,1)

    # 注意:f1计算用raw_data,而不是target
    # acc虽然保留，但用于衡量NER任务不合适

    with torch.no_grad():
        for i, (data, target, word) in enumerate(loader):
            #data, target = data.to(args.device), target.to(args.device)
            data = data.to(args.device)
            target = convert_bio_to_io_idx(target).to(args.device)
            # compute output
            output = model(data)
            loss = criterion(output, target)
            loss = loss.mean()
            # # measure accuracy and record loss
            test_loss.update(loss.item(), data.size(0))
            acc1 = compute_topk_accuracy(output, target, topk=(1,))
            correct.update(acc1[0].item(), data.size(0))
            words.extend(word)
            predictions += torch.argmax(output,dim=1).cpu().tolist()
            #print(target)
            #print(predictions)
    
    predictions = label_representation.predictions_to_labels(predictions)
    # if predictions are in IO format, convert to BIO used for evaluation when working on test set
    if TRAIN_SETTINGS["LABEL_FORMAT"] == "io":
        predictions = LabelRepresentation.convert_io_to_bio_labels(predictions)
    evaluation = Evaluation(separator=TRAIN_SETTINGS["DATA_SEPARATOR"])
    connl_evaluation_string = evaluation.create_connl_evaluation_format(raw_data, words, predictions)
    evaluation_output = evaluation.evaluate_evaluation_string(connl_evaluation_string)
    f1 = Evaluation.extract_f_score(evaluation_output)
    
    log(logpath, 'Time for {}-Epoch-{}/{}:{:.1f}s Acc:{}, Loss:{}, F1:{}\n'.format('Test'if mode=='test'else 'Val',
                      epoch, args.epochs, time.time()-t0, correct.avg, test_loss.avg, f1))
    log_value('{}/loss'.format(mode), test_loss.avg, step=epoch)
    # Logging results on tensorboard
    log_value('{}/accuracy'.format(mode), correct.avg, step=epoch)

    log_value('{}/f1'.format(mode), f1, step=epoch)
    return test_loss.avg, correct.avg, f1, evaluation_output

def main(params):
    """Objective function for Hyperparameter Optimization"""
    # Keep track of evals
    
    if 'STGN' in args.exp_name:
        #TODO: automatic adjustment (sig_max, lr_sig)
        args.times = params['times']
        args.sigma = params['sigma']
        args.sig_max = 2.0 * params['sigma']
        args.lr_sig = 0.1 * params['sigma']
        #others
        args.avg_steps = params['avg_steps']
        args.ratio_l = params['ratio_l']
        args.noise_rate = params['noise_rate'] #这里应该由脚本指定
        args.forget_times = params['forget_times']
    
    if 'SLN' in args.exp_name:
        args.sigma = params['sigma']

    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)
    args.logpath = os.path.join(args.exp_name, 'log.txt')
    args.log_dir = os.path.join(os.getcwd(), args.exp_name)
    
    # 暂时注释，等会加回来
    generate_log_dir(args)
    #should be placed after generate_log_dir()
    log(args.logpath, 'Settings: {}\n'.format(args))

    args.device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu_id)
    set_seed(args)

    loaders, mdl_loss = MD_CLASSES[args.dataset]

    train_loader, dev_loader, test_loader, train_noisy, dev, test, noisy_ind, clean_ind, label_representation = loaders(args)

    # Create model
    net, criterion, criterion_val = mdl_loss(args)

    #train_loader, test_loader, train_dataset, test_dataset, noisy_ind, clean_ind = loaders(args)

    #update perturb variance, dynamic sigma for each sample
    #暂时未写入文件保存
    args.sigma_dyn = torch.tensor([args.sigma]*len(train_noisy),
                           dtype=torch.float32,
                           requires_grad=False,
                           device=args.device)

    args.prev_acc = torch.tensor(np.zeros(len(train_noisy)),
                           dtype=torch.long,
                           requires_grad=False,
                           device=args.device)
    args.forgetting = torch.tensor(np.zeros(len(train_noisy)),
                                 dtype=torch.long,
                                 requires_grad=False,
                                 device=args.device)

    cudnn.benchmark = True

    epochs = TRAIN_SETTINGS["EPOCHS"]
    lr = TRAIN_SETTINGS["LEARNING_RATE"]

    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    args.epochs = epochs

    # Training
    global_t0 = time.time()
    global_iter = 0
    global val_best, test_best
    val_best, test_best = 0, 0
    test_best_str = ""
    res_lst = []

    args.drop_rate_schedule = np.ones(args.epochs) * args.noise_rate
    args.drop_rate_schedule[:args.num_gradual] = np.linspace(0, args.noise_rate, args.num_gradual)


    for epoch in range(1, args.epochs + 1):
        args.cur_epoch = epoch
        global_iter, train_loss, train_acc, tc_acc, fc_acc = train_others(args, net, train_loader, optimizer, criterion,
                                                            global_iter, epoch, args.logpath)

        val_loss, val_acc, val_f1, val_str = validate(args, net, dev_loader, criterion_val, epoch, args.logpath, dev, label_representation, mode='val')
        test_loss, test_acc, test_f1, test_str = validate(args, net, test_loader, criterion_val, epoch, args.logpath, test, label_representation, mode='test')
        
        logging.info(f'Epoch {epoch + 1}\tCurrent F1 for DEV: {val_f1}\tTEST: {test_f1}')

        # Save checkpoint.
        if val_f1 > val_best:
            val_best = val_f1
            test_best = test_f1
            test_best_str = test_str
            # TODO 调参的时候，没有保存模型，节省存储空间
            # utils.checkpoint(val_acc, epoch, model, args.save_dir)

        res_lst.append((train_acc, tc_acc, fc_acc, test_acc, test_f1, test_best, train_loss, test_loss))
        if len(noisy_ind)>0:
            log_stats(data=torch.tensor([args.sigma_dyn[i] for i in noisy_ind]),
                    name='epoch_stats_sigma_dyn_noisy',
                    step=epoch)
        if len(clean_ind)>0:
            log_stats(data=torch.tensor([args.sigma_dyn[i] for i in clean_ind]),
                    name='epoch_stats_sigma_dyn_clean',
                    step=epoch)

    run_time = time.time()-global_t0
    #save 3 types of acc
    with open(os.path.join(args.log_dir, 'acc_loss_results.txt'), 'w', newline='') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerows(res_lst)
    
    # Val_best Test_at_val_best
    with open(os.path.join(args.log_dir, 'best_results.txt'), 'w') as outfile:
        outfile.write(f'{val_best}\t{test_best}\n{test_best_str}')
        
    log(args.logpath, '\nBest F1: {}\n{}'.format(test_best,test_best_str))
    log(args.logpath, '\nTotal Time: {:.1f}s.\n'.format(run_time))
    logging.info(f'best_dev: {val_best}, best_dev_test: {test_best}')
    logging.info(test_best_str)

    # loss = - val_best
    loss = - test_best
    return {'loss': loss, 'best_f1': val_best, 'test_at_best': test_best,
            'params': params, 'train_time': run_time, 'status': STATUS_OK}

if __name__ == '__main__':
    print("load params from : ", args.params_path)
    # TODO 之前的方式是载入 ['best']，这里更换是因为程序跑的总是提前停机，产生不了 best params 文件，所以手动选择要载入的参数选择文件
    params = json.load(open(args.params_path, 'r', encoding="utf-8"))['best']  if 'base' not in args.exp_name else {}
    assert params is not None
    main(params=params)