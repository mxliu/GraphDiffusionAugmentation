import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import math
import time
import shutil
import pretext_model
from diffusion_loss import TrainLoss
from dataset import X1,E1,y


parser = argparse.ArgumentParser(description='PyTorch MDD Training')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--seed', default='', type=int,
                    help='seed for initializing training. ')

# model specific configs:
parser.add_argument('--T', default=1000, type=int,
                    help='time steps')   
parser.add_argument('--diffusion_hidden_mlp_dims', default={'X': 64, 'E': 4, 'y': 16}, type=dict,
                    help=' hidden_mlp_dims of transformer in diffusion model') 
parser.add_argument('--diffusion_hidden_dims', default={'dx': 64, 'de': 8, 'dy': 8, 'n_head': 2, 'dim_ffX': 128, 'dim_ffE': 16, 'dim_ffy': 16}, 
                    type=dict,help=' hidden_dims of transformer in diffusion model') 
parser.add_argument('--diffusion_num_layers', default=1, type=int,
                    help='diffusion_num_layers') 

parser.add_argument('--GIN_input_dim', default=116, type=int,
                    help='GIN_input_dim')  
parser.add_argument('--GIN_hidden_dim', default=64, type=int,
                    help='GIN_hidden_dim')  
parser.add_argument('--GIN_num_layers', default=2, type=int,
                    help='GIN_num_layers') 
                  
parser.add_argument('--projector_input_dim', default=64, type=int,
                    help='feature dimension (default: 64)')
parser.add_argument('---projector_hidden_dim', default=32, type=int,
                    help='hidden dimension of the projector (default: 32)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the projector')

args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)                                           
    cudnn.deterministic = True 



def train(train_loader, Model, criterion1, criterion2, optimizer, epoch, args):
    # calculate and store averages and current values
    batch_time = AverageMeter('Time', ':6.3f')   
    data_time = AverageMeter('Data', ':6.3f')   
    losses = AverageMeter('Loss', ':.4f')
    # used for outputting intermediate training results in real time
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode 
    Model.train()
    #time of program execution
    end = time.time()   #
    for i,(X1_batch, E1_batch, y_batch) in enumerate(train_loader):
        # measure data loading time  
        data_time.update(time.time() - end)
        
        X1_batch = X1_batch.float().to(device)
        E1_batch = E1_batch.float().to(device)
        y_batch = y_batch.float().to(device)
        
        pred,p1, p2, z1, z2 = Model(X1_batch, E1_batch, y_batch)
        
        #diffusion loss
        loss1 = criterion1(pred.X, pred.E, pred.y, X1_batch, E1_batch, y_batch, True)
        #contrastive loss
        loss2 = -(criterion2(p1, z2).mean() + criterion2(p2, z1).mean()) * 0.5
        #objective function
        loss=loss1+loss2
              
        losses.update(loss.item(), X1_batch.size(0))   

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # measure elapsed time 
        batch_time.update(time.time() - end)
        end = time.time() 
       
        if i % args.print_freq == 0:
            progress.display(i)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = args.lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
           param_group['lr'] = args.lr
        else:
           param_group['lr'] = cur_lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


class ProgressMeter(object):
    """Assist intermediate result output"""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


# create model    
diffusion_output_dims = {'X': X1.size(-1),'E': E1.size(-1),'y': 0} 

Model = pretext_model.Pretext_model(device=device,T=args.T,diffusion_hidden_mlp_dims=args.diffusion_hidden_mlp_dims,
                                    diffusion_hidden_dims=args.diffusion_hidden_dims,diffusion_output_dims=diffusion_output_dims,
                                    diffusion_num_layers=args.diffusion_num_layers, GIN_input_dim=args.GIN_input_dim, 
                                    GIN_hidden_dim=args.GIN_hidden_dim, GIN_num_layers=args.GIN_num_layers, 
                                    projector_input_dim=args.projector_input_dim, projector_hidden_dim=args.projector_hidden_dim)

Model.to(device)

# define loss function (criterion) and optimizer
criterion1=TrainLoss(device=device)
criterion2 = nn.CosineSimilarity()

if args.fix_pred_lr:
    optim_params = [{'params': Model.encoder.parameters(), 'fix_lr': False},
                    {'params': Model.predictor.parameters(), 'fix_lr': True}]
else:
    optim_params = Model.parameters()

optimizer = torch.optim.SGD(optim_params, args.lr,momentum=args.momentum,
                            weight_decay=args.weight_decay)
#optimizer = torch.optim.Adam(optim_params,args.lr,weight_decay=args.weight_decay)


#Continue the previously saved training progress
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
            
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        Model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True



train_dataset=torch.utils.data.TensorDataset(X1,E1,y)
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

training_loss = 0.0 

for epoch in range(args.start_epoch, args.epochs):
    
    adjust_learning_rate(optimizer, epoch, args)
    train(train_loader, Model, criterion1, criterion2, optimizer, epoch, args)
    if (epoch+1) % 10 == 0:
       save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': Model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint100_{:03d}_1.pth.tar'.format(epoch))


 







  


