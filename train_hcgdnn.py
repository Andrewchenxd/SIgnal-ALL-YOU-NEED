import numpy as np
import torch.optim
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from model.HydraAttention_dropout_RMLdrop import *
from model.HydraAttention_cutmix_dropout_RMLdrop import  *
from dataset import *
from tqdm import tqdm
import torch.nn.functional as F
from utils.utils import *
import os
import sys
from model.mcldnn import mcldnn
from model.resnet1d import ResNet1d
from model.resnet2d import ResNet2d
from model.lstm import lstm2
from model.CNN2 import CNN2
from model.DAE import DAE
from model.HCGDNN import HCGDNN
from scipy.optimize import Bounds,minimize,LinearConstraint
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
choose=True
import argparse
def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False
if choose==True:
    parser = argparse.ArgumentParser(description='Train TransNet')
    # parser.add_argument("--datapath", type=str, default='./20classes_8.28/mix_123.mat')
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--classesnum", type=int, default=11)
    parser.add_argument("--netdepth", type=int, default=64)
    parser.add_argument("--cutmixsize", type=int, default=4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--wait", type=int, default=10)
    parser.add_argument("--declay", default=0.5)
    parser.add_argument("--yuzhi", type=int, default=5)
    parser.add_argument("--numworks", type=int, default=2)
    parser.add_argument("--pref", type=int, default=20)
    parser.add_argument("--trans_choose", type=str, default='pwvd')
    # parser.add_argument("--dataset", type=str, default='adsb')
    # parser.add_argument("--name", type=str, default='adsb')
    parser.add_argument("--dataset", type=str, default='RML2016.10a')
    parser.add_argument("--name", type=str, default='RML2016.10a')
    parser.add_argument("--withoutis", type=str, default='no')
    parser.add_argument("--adsbis", type=str2bool, default=False)
    parser.add_argument("--resample", type=str2bool, default=False)
    parser.add_argument("--chazhi", type=str2bool, default=False)
    parser.add_argument("--newdata", type=str2bool, default=False)
    parser.add_argument("--is_DAE", type=str2bool, default=False)
    parser.add_argument("--cnum", type=int, default=2)
    parser.add_argument("--samplenum", type=int, default=2)  # samplenum pwvd 15 without 5
    opt = parser.parse_args()

def acc_classes(pre, labels,BATCH_SIZE):
    pre_y = torch.max(pre, dim=1)[1]
    train_acc = torch.eq(pre_y, labels.to(device)).sum().item() / BATCH_SIZE
    return train_acc

def acc_AA(pre, labels,acc_AA_pre,acc_AA_count):
    pre_y=torch.max(pre, dim=1)[1]
    pre_y = pre_y.detach().cpu().numpy()
    labelclass=np.array(labels)
    # labelclass[labelclass == 99] = 7
    for i in range(len(labelclass)):
        if pre_y[i]==labelclass[i]:
            acc_AA_pre[0,labelclass[i]]+=1
            acc_AA_count[0,labelclass[i]]+=1
        else:
            acc_AA_count[0, labelclass[i]] += 1
    return acc_AA_pre,acc_AA_count

def acc_snrs(pre, labels,snr,acc_snr_pre,acc_snr_count):
    pre_y = torch.max(pre, dim=1)[1]
    pre_y =pre_y.detach().cpu().numpy()
    labelclass=np.array(labels)
    for i in range(len(labelclass)):
        if pre_y[i]==labelclass[i]:
            acc_snr_pre[0,snr[i]]+=1
            acc_snr_count[0,snr[i]]+=1
        else:
            acc_snr_count[0, snr[i]] += 1
    return acc_snr_pre,acc_snr_count

def funsub(p1,p2,p3):
    p = p1+p2+p3
    c=745
    pmax=np.max(p,axis=1)
    pmax=np.expand_dims(pmax,axis=1)
    psum=np.sum(np.exp((p-pmax)*c),axis=1)
    psum=np.expand_dims(psum,axis=1)
    v=np.exp((p-pmax)*c)/psum
    return v

# demo 2
# 计算  (2+x1)/(1+x2) - 3*x1+4*x3 的最小值  x1,x2,x3的范围都在0.1到0.9 之间
def fun(args):
    p1, p2, p3, t = args
    # v = lambda x: np.mean(((x[0]*p1+x[1]*p2+x[2]*p3)-t)**2)
    v = lambda x: np.mean((funsub(x[0]*p1,x[1]*p2,x[2]*p3) - t) ** 2)
    return v
def toonehot(t):
    b=t.shape[0]
    tnew=np.zeros((b,args.classesnum))
    for i in range(b):
        tnew[i,t[i]]=1
    return tnew

def train(train_loader, model, criterion1,criterion2, optimizer, epoch, epoch_max,batchsize,adsbis=False):
    """Train for one epoch on the training set"""
    losses_class = AverageMeter()
    acc = AverageMeter()
    if adsbis==True:
        acc_snr_pre=np.zeros((1,7))
        acc_snr_count = np.zeros((1,7))
    else:
        acc_snr_pre = np.zeros((1, 20))
        acc_snr_count = np.zeros((1, 20))
    w1r,w2r,w3r=0,0,0
    # switch to train mode
    model.train()
    with tqdm(total=len(train_loader), desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3) as pbar:
        for i, (input1,input2,input3, target,snr) in enumerate(train_loader):
            images, sgn, te, labels = input1, input2, input3, target

            output1,output2,output3= model(images.to(device),sgn.to(device),te.to(device))
            target_var=labels.to(device)
            # target_var = target_var.to(torch.float)
            loss1=criterion2(output1,target_var)
            loss2 = criterion2(output2, target_var)
            loss3 = criterion2(output3, target_var)
            # loss=loss1+loss2+loss3
            p1,p2,p3=F.softmax(output1.detach(), dim=1),F.softmax(output2.detach(), dim=1),F.softmax(output3.detach(), dim=1)
            p1,p2,p3=p1.detach().cpu().numpy(),p2.detach().cpu().numpy(),p3.detach().cpu().numpy()
            t=target.detach().cpu().numpy()
            t=toonehot(t)
            args = (p1, p2, p3, t)
            A = [[1, 1, 1], ]
            lb1 = [1, ]
            ub1 = [1, ]
            linear_cosnet = LinearConstraint(A, lb1, ub1)
            # 设置初始猜测值
            x0 = np.asarray((0.0, 0.0, 1))
            lb = [0, 0, 0]
            ub = [1, 1, 1]
            bounds = Bounds(lb, ub)

            # 关闭print的输出
            sys.stdout = open(os.devnull, 'w')
            w1,w2,w3 = minimize(fun(args), x0, method='trust-constr', constraints=linear_cosnet, bounds=bounds,
                           options={'verbose': 1}).x
            # w1,w2,w3=res.x
            sys.stdout = sys.__stdout__
            w1r+=w1
            w2r+=w2
            w3r+=w3
            out=w1*output1+w2*output2+w3*output3
            loss =  w1*loss1 + w2*loss2 + w3*loss3
            # measure accuracy and record loss
            acc.update(acc_classes(out.data, target,batchsize))
            if adsbis == True:
                acc_snrs(out, labels, snr-1, acc_snr_pre, acc_snr_count)
            else:
                acc_snrs(out, labels, snr, acc_snr_pre, acc_snr_count)
            losses_class.update(loss.item())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(**{'train_loss_': losses_class.avg,
                                'acc': acc.avg})
            pbar.update(1)
    print(acc_snr_pre/acc_snr_count*100)
    w1r=w1r/i
    w2r=w2r/i
    w3r = w3r / i
    return acc.avg, losses_class.avg,w1r,w2r,w3r


def validate(val_loader, model, criterion1, criterion2,epoch, epoch_max,batchsize,w1,w2,w3,adsbis=False):
    """Perform validation on the validation set"""
    losses_class = AverageMeter()
    acc = AverageMeter()
    if adsbis == True:
        acc_snr_pre_val = np.zeros((1, 7))
        acc_snr_count_val = np.zeros((1, 7))
    else:
        acc_snr_pre_val = np.zeros((1, 20))
        acc_snr_count_val = np.zeros((1, 20))
    counttime = 0
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3,
                  colour='blue') as pbar:
            for i, (input1,input2,input3, target,snr) in enumerate(val_loader):
                val_image, val_sgn, val_te, val_label = input1, input2, input3, target
                s = time.time()
                output1,output2,output3= model(val_image.to(device), val_sgn.to(device), val_te.to(device))
                e = time.time()
                counttime += (e - s)
                target_var = val_label.to(device)
                # target_var = target_var.to(torch.float)
                loss1 = criterion2(output1, target_var)
                loss2 = criterion2(output2, target_var)
                loss3 = criterion2(output3, target_var)
                # loss = loss1 + loss2 + loss3
                loss =  w1*loss1 + w2*loss2 + w3*loss3
                out=w1*output1+w2*output2+w3*output3
                # out=out/3
                # measure accuracy and record loss
                acc.update(acc_classes(out.data, target, batchsize))
                if adsbis == True:
                    acc_snrs(out, val_label, snr-1, acc_snr_pre_val, acc_snr_count_val)
                else:
                    acc_snrs(out, val_label, snr, acc_snr_pre_val, acc_snr_count_val)

                losses_class.update(loss.item())

                pbar.set_postfix(**{'val_loss_class': losses_class.avg,
                                    'acc': acc.avg})
                pbar.update(1)
    print(acc_snr_pre_val / acc_snr_count_val * 100)
    print(counttime)
    return acc.avg, losses_class.avg


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.trans_choose == "stft":
        train_set = SigDataSet_stft('./data622/{}_train_data.mat'.format(args.dataset), newdata=args.newdata,
                                    adsbis=args.adsbis
                                    , resample_is=args.resample, samplenum=args.samplenum)
        # train_set_gao = SigDataSet_stft('./data622/{}_train_gaodata.mat'.format(args.dataset))
        # train_set_gao5 = SigDataSet_stft('./data622/{}_train_snr(da5)data.mat'.format(args.dataset))
        test_set = SigDataSet_stft('./data622/{}_test_data.mat'.format(args.dataset), newdata=args.newdata,
                                   adsbis=args.adsbis
                                   , resample_is=args.resample, samplenum=args.samplenum)
        if args.dataset == 'RML2016.10a':
            val_set = SigDataSet_stft('./data622/{}_test170_data.mat'.format(args.dataset), newdata=args.newdata,
                                      adsbis=args.adsbis
                                      , resample_is=args.resample, samplenum=args.samplenum)
        else:
            val_set = SigDataSet_stft('./data622/{}_val_data.mat'.format(args.dataset), newdata=args.newdata,
                                      adsbis=args.adsbis
                                      , resample_is=args.resample, samplenum=args.samplenum)
    elif args.trans_choose == "pwvd":
        train_set = SigDataSet_pwvd('./data622/{}_train_data.mat'.format(args.dataset), newdata=args.newdata,
                                    adsbis=args.adsbis,
                                    resample_is=args.resample, samplenum=args.samplenum, is_DAE=args.is_DAE)
        # train_set_gao = SigDataSet_pwvd('./data622/{}_train_gaodata.mat'.format(args.dataset), newdata=args.newdata,
        #                                 adsbis=args.adsbis,
        #                                 resample_is=args.resample, samplenum=args.samplenum, is_DAE=args.is_DAE)
        # train_set_gao5 = SigDataSet_pwvd('./data622/{}_train_snr(da5)data.mat'.format(args.dataset))
        # train_set_gao2 = SigDataSet_pwvd('./data622/{}_train_snr(da2_9)data.mat'.format(args.dataset))
        test_set = SigDataSet_pwvd('./data622/{}_test_data.mat'.format(args.dataset), newdata=args.newdata,
                                   adsbis=args.adsbis,
                                   resample_is=args.resample, samplenum=args.samplenum, is_DAE=args.is_DAE)
        if args.dataset == 'RML2016.10a':
            val_set = SigDataSet_pwvd('./data622/{}_test170_data.mat'.format(args.dataset), newdata=args.newdata,
                                      adsbis=args.adsbis,
                                      resample_is=args.resample, samplenum=args.samplenum, is_DAE=args.is_DAE)
        else:
            val_set = SigDataSet_pwvd('./data622/{}_val_data.mat'.format(args.dataset), newdata=args.newdata,
                                      adsbis=args.adsbis,
                                      resample_is=args.resample, samplenum=args.samplenum, is_DAE=args.is_DAE)
    elif args.trans_choose == "spwvd":
        train_set = SigDataSet_spwvd('./data622/{}_train_data.mat'.format(args.dataset))
        train_set_gao = SigDataSet_spwvd('./data622/{}_train_gaodata.mat'.format(args.dataset))
        train_set_gao5 = SigDataSet_spwvd('./data622/{}_train_snr(da5)data.mat'.format(args.dataset))
        train_set_gao2 = SigDataSet_spwvd('./data622/{}_train_snr(da2_9)data.mat'.format(args.dataset))
        if args.dataset == 'RML2016.10a':
            val_set = SigDataSet_spwvd('./data622/{}_test170_data.mat'.format(args.dataset))
        else:
            val_set = SigDataSet_spwvd('./data622/{}_val_data.mat'.format(args.dataset))
        test_set = SigDataSet_spwvd('./data622/{}_test170_data.mat'.format(args.dataset))
    train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=args.numworks
                              , prefetch_factor=args.pref,drop_last=True)
    # train_loader_gao = DataLoader(train_set_gao, batch_size=args.batchsize, shuffle=True, num_workers=14
    #                               , prefetch_factor=args.pref)
    # train_loader_gao5 = DataLoader(train_set_gao5, batch_size=args.batchsize, shuffle=True, num_workers=args.numworks
    #                                , prefetch_factor=args.pref)
    val_loader = DataLoader(val_set, batch_size=args.batchsize, num_workers=args.numworks
                            , prefetch_factor=args.pref, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batchsize, num_workers=args.numworks
                             , prefetch_factor=args.pref, shuffle=True,drop_last=True)
    if args.withoutis == "no":
        # model = sknetdrop(args.classesnum, args.netdepth, args.cutmixsize,in_features=2368)
        model=HCGDNN(args.classesnum)
        print("use normal model")
    else:
        model = sknetdrop_withoutcutmix(args.classesnum, args.netdepth, args.cutmixsize,drop=0.4)
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = nn.DataParallel(model)
        model = model.to(device)
    else:
        model = model.cuda()
    # state_dict = torch.load(
    #     './checkpoint_othernets_adsb/resample2/HCGDNN/pwvd_best_network_acc_0.5217633928571429.pth')
    # model.load_state_dict(state_dict)
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    #optimizer = optim.AdamW(model.parameters(), lr=args.lr)  # 学习率0.001
    optimizer_sgd = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.005,
        nesterov=True
    )
    optimizer_adam = torch.optim.Adam(model.parameters(),lr=args.lr)
    optimizer = optimizer_adam
    csv_logger = CSVStats()
    early_stopping = EarlyStopping_acc(save_path='./checkpoint_othernets_{}/HCGDNN/test'.format(args.dataset), patience=args.patience,
                                       wait=args.wait, choose=args.trans_choose
                                       , best_score=0.521)
    wait_idem = args.wait
    declay_count = 0
    acc_train=0
    loss_train=0
    w1,w2,w3=0.5,0.25,0.25
    for epoch in range(0, args.epochs):

        # acc_train, loss_train = train(
        #     train_loader_gao, model, criterion, optimizer, epoch, batchsize=args.batchsize, epoch_max=args.epochs)
        #
        # torch.cuda.empty_cache()

        acc_train, loss_train,w1,w2,w3 = train(
            train_loader, model, criterion1, criterion2, optimizer, epoch, batchsize=args.batchsize, epoch_max=args.epochs,adsbis=args.adsbis)

        torch.cuda.empty_cache()

        acc_val, loss_val = validate(
            val_loader, model, criterion1, criterion2, epoch, batchsize=args.batchsize, epoch_max=args.epochs,adsbis=args.adsbis,w1=w1,w2=w2,w3=w3)

        # Print some statistics inside CSV
        csv_logger.add(acc_train, acc_val, loss_train, loss_val,args.lr)
        csv_logger.write(patience=args.patience,wait=args.wait,choose=args.trans_choose,name=args.name)

        early_stopping(acc_val, model)
        if early_stopping.flag==True:
            wait_idem=args.wait
        if early_stopping.counter >5:
            wait_idem += 1
            if wait_idem>=args.wait:
                args.lr = adjust_learning_rate(optimizer, args.lr,args.declay)
                wait_idem=0
                declay_count+=1
            if  declay_count>=args.yuzhi:
                args.lr = adjust_learning_rate(optimizer, 0.001*(0.5)**3, args.declay)
                declay_count = 0
        print(args.wait)
        if early_stopping.early_stop:
            print("Early stopping")
            break
