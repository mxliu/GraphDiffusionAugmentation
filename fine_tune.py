import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import math
import torch.utils.data
from sklearn.metrics import confusion_matrix
import fine_tune_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve,auc,balanced_accuracy_score
from monai.utils import set_determinism


parser = argparse.ArgumentParser(description='PyTorch MDD Finetne')

parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)') 

parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-b1', '--batch-size1', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-b2', '--batch-size2', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')          
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')                   

parser.add_argument('--seed', default='', type=int,
                     help='seed for initializing training. ')


parser.add_argument('--GIN_input_dim', default=116, type=int,
                    help='GIN_input_dim')  
parser.add_argument('--GIN_hidden_dim', default=64, type=int,
                    help='GIN_hidden_dim')  
parser.add_argument('--GIN_num_layers', default=2, type=int,
                    help='GIN_num_layers') 
parser.add_argument('--dim', default=64, type=int,
                    help='feature dimension (default: 64)')
parser.add_argument('--pred-dim', default=32, type=int,
                    help='hidden dimension of the predictor (default: 32)')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')              

args = parser.parse_args()


if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    set_determinism(seed=args.seed)



def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = args.lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

#how to train in each epoch
def train(finetune_loader, model, criterion, optimizer, epoch):   

    model.train()
    node_feature_batch=None
    for i,(finetune_X,finetune_E,finetune_label) in enumerate(finetune_loader):
        node_feature,z,output= model(finetune_X,finetune_E)
        
        node_feature = node_feature.data.cpu().numpy() 
        if node_feature_batch is None:              
            node_feature_batch = node_feature
        else:
            node_feature_batch = np.concatenate((node_feature_batch, node_feature), axis=0)
 
        output=output.to(torch.float32)
        finetune_label=finetune_label.to(torch.float32)  
        loss = criterion(output, finetune_label.long())

        output = output.data.cpu().numpy() 
        y_pred = np.around(output,0).astype(int)
        y_pred = torch.from_numpy(y_pred)
    
        finetune_label = finetune_label.data.cpu().numpy()
        y_pred = torch.argmax(y_pred, dim=1)
        finetune_acc=accuracy_score(finetune_label, y_pred) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
       
    
        if (epoch+1) % 10 == 0:
       
          print('[%d]' % (epoch + 1))
          print('output:',output)
          print('y_pred:',y_pred)   
           
    return loss,finetune_acc,node_feature_batch
    

 

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

criterion=nn.CrossEntropyLoss()

                                                   

def calculate_metric(gt, pred):
    pred[pred > 0.5] = 1
    pred[pred < 1] = 0
    confusion = confusion_matrix(gt, pred)
   
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]   
    
    spe = TN / float(TN + FP)
     
    npv = TN/float(TN+FN) 
   
    return spe,npv

epoch_loss=0.0

acc=0.0
pre=0.0
rec=0.0
f1=0.0
bac=0.0
AUC=0.0
spe=0.0
npv=0.0
acc_all=None
auc_all=None
sen_all=None
spe_all=None
ppv_all=None
npv_all=None
f1_all =None
bac_all=None

for fold in range(1,6):
    
    print('fold:',fold)
    
    # load data
    finetune_X = np.load('./finetune_X_'+str(fold)+'.npy') 
    finetune_E = np.load('./finetune_E_'+str(fold)+'.npy')
    finetune_label = np.load('./finetune_label_'+str(fold)+'.npy') 
    
    test_X = np.load('./test_X_'+str(fold)+'.npy') 
    test_E = np.load('./test_E_'+str(fold)+'.npy')
    test_label = np.load('./test_label_'+str(fold)+'.npy') 
    
   
    finetune_X = torch.from_numpy(finetune_X).float().to(device)
    finetune_E = torch.from_numpy(finetune_E).float().to(device)
    finetune_label = torch.from_numpy(finetune_label).float().to(device)
    finetune_dataset = torch.utils.data.TensorDataset(finetune_X, finetune_E, finetune_label)
    finetune_loader = torch.utils.data.DataLoader(finetune_dataset,batch_size=args.batch_size1, num_workers=0, shuffle=True)
    
    test_X = torch.from_numpy(test_X).float().to(device)
    test_E = torch.from_numpy(test_E).float().to(device)
    test_label=torch.from_numpy(test_label).float().to(device)
    test_dataset=torch.utils.data.TensorDataset(test_X,test_E,test_label)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size2,  num_workers=0,shuffle=True)
    
    # create model
    model = fine_tune_model.Task_specific_model(GIN_input_dim=args.GIN_input_dim,
                                                GIN_hidden_dim=args.GIN_hidden_dim,
                                                GIN_num_layers=args.GIN_num_layers,
                                                dim=args.dim, 
                                                pred_dim=args.pred_dim)
    model.to(device)


        
    # init the fc layer  
    model.predictor.L1.weight.data.normal_(mean=0.0, std=0.01)
    model.predictor.BN.weight.data.normal_(mean=0.0, std=0.01)
    model.predictor.L2.weight.data.normal_(mean=0.0, std=0.01)
    
    model.predictor.BN.bias.data.zero_()
    model.predictor.L2.bias.data.zero_()
   
    
    
    #encoder parameter initialization
    if args.pretrained:
        if os.path.isfile(args.pretrained):                                    
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")       #load pre-trained model parameters
    
            # rename pre-trained keys  
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer  
                if k.startswith('base_encoder'):
                    # remove prefix
                    state_dict[k[len("base_"):]] = state_dict[k]
                    # delete renamed or unused k   
                del state_dict[k]
                
    
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            print(set(msg.missing_keys))
            assert set(msg.missing_keys) == {"predictor.L1.weight","predictor.BN.weight",
                                             "predictor.L2.weight","predictor.BN.bias",
                                             "predictor.L2.bias","predictor.BN.running_mean",
                                             "predictor.BN.num_batches_tracked","predictor.BN.running_var"}          
    
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
        
    #parameters = list(filter(lambda p: p.requires_grad, model.parameters())) 
        
    optimizer = torch.optim.Adam(model.parameters(),args.lr,weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(parameters, args.lr,
    #                              momentum=args.momentum,
    #                              weight_decay=args.weight_decay)

    
    

    #start training each fold
    loss_fold=None
    for epoch in range(args.start_epoch, args.epochs):
        
        
        adjust_learning_rate(optimizer, epoch, args)
        
        loss,finetune_acc,node_feature_batch=train(finetune_loader,model, criterion, optimizer, epoch)
        
        
        
        if epoch+1 == args.epochs:
           
           filename = './node_feature_'+str(fold)
           np.save(filename,node_feature_batch)
           
           filename = './loss_fold_'+str(fold)
           np.save(filename,loss_fold)
           
               
        
        if (epoch+1) % 10 == 0:    
           print('finetune_acc:%f' % finetune_acc)
        
        
        if epoch+1 == args.epochs:
            #model.eval()
            with torch.no_grad():
            
                feature_batch=None
                prediction_batch=None
                pred_batch=None
                test_label_batch=None 
                node116_feature_batch=None
                for i,(test_X,test_E,test_label) in enumerate(test_loader):
                    
                    node116_feature,feature,prediction= model(test_X,test_E )
                    
                    node116_feature = node116_feature.data.cpu().numpy() 
                    if node116_feature_batch is None:              
                       node116_feature_batch = node116_feature
                    else:
                       node116_feature_batch = np.concatenate((node116_feature_batch, node116_feature), axis=0)
                        
                    feature = feature.data.cpu().numpy() 
                    if feature_batch is None:              
                        feature_batch = feature
                    else:
                        feature_batch = np.concatenate((feature_batch, feature), axis=0)  
                        
                    prediction = prediction.data.cpu().numpy() 
                    if prediction_batch is None:              
                        prediction_batch = prediction
                    else:
                        prediction_batch = np.concatenate((prediction_batch, prediction), axis=0)
                     
                  
                    pred = np.around(prediction,0).astype(int)
                    if pred_batch is None:              
                        pred_batch = pred
                    else:
                        pred_batch = np.concatenate((pred_batch, pred), axis=0)
                           
                  
                    test_label = test_label.data.cpu().numpy()
                    if test_label_batch is None:              
                        test_label_batch = test_label
                    else:
                        test_label_batch = np.concatenate((test_label_batch, test_label), axis=0)
                       
                #for drawing and calculating  results
                filename = './node116_feature'+str(fold)
                np.save(filename,node116_feature_batch)          
            
                filename = './feature_'+str(fold)
                np.save(filename,feature_batch)
                             
                print('prediction:',prediction_batch)
                filename = './prediction_'+str(fold)
                np.save(filename,prediction_batch)
                             
                pred_batch = torch.from_numpy(pred_batch)
                pred_batch = torch.argmax(pred_batch, dim=1)
                print('pred:',pred_batch)
                                
                filename = './label_'+str(fold)
                np.save(filename,test_label_batch)
                
                fold_acc=accuracy_score(test_label_batch, pred_batch)
                print('fold_acc:',fold_acc)
                
                fold_pre=precision_score(test_label_batch,pred_batch)
                print("fold_pre:", fold_pre)
                
                fold_rec=recall_score(test_label_batch,pred_batch)
                print("fold_rec:", fold_rec)
                
                fold_f1=f1_score(test_label_batch,pred_batch)
                print("fold_f1:", fold_f1)
                
                fold_bac=balanced_accuracy_score(test_label_batch,pred_batch)
                print("fold_bac:",fold_bac)
                
                fpr, tpr, thresholds = roc_curve(test_label_batch,prediction_batch[:,1])
                fold_auc = auc(fpr, tpr)
                print("fold_auc:",fold_auc)
                
                fold_spe,fold_npv=calculate_metric(test_label_batch,pred_batch)
                print("fold_spe:",fold_spe)
                print("fold_npv:",fold_npv)

                     
    fold_acc = np.array(fold_acc)  
    fold_pre = np.array(fold_pre)
    fold_rec = np.array(fold_rec)
    fold_f1 = np.array(fold_f1)
    fold_bac = np.array(fold_bac)
    fold_auc = np.array(fold_auc)
    fold_spe = np.array(fold_spe)
    fold_npv = np.array(fold_npv)
    
    acc += fold_acc.item()
    pre += fold_pre.item()
    rec += fold_rec.item()
    f1 += fold_f1.item()
    bac += fold_bac.item()
    AUC+=fold_auc.item()
    spe+=fold_spe.item()
    npv+=fold_npv.item()
    
    
    if acc_all is None and auc_all is None and sen_all is None and sen_all is None and spe_all is None and ppv_all is None and ppv_all is None and  f1_all is None and bac_all is None:
      acc_all=[fold_acc]
      auc_all=[fold_auc]
      sen_all=[fold_rec]
      spe_all=[fold_spe]
      ppv_all=[fold_pre]
      npv_all=[fold_npv]
      f1_all =[fold_f1]
      bac_all=[fold_bac]
      
    else:
      acc_all=np.concatenate((acc_all,[fold_acc])) 
      auc_all=np.concatenate((auc_all,[fold_auc])) 
      sen_all=np.concatenate((sen_all,[fold_rec])) 
      spe_all=np.concatenate((spe_all,[fold_spe])) 
      ppv_all=np.concatenate((ppv_all,[fold_pre]))
      npv_all=np.concatenate((npv_all,[fold_npv])) 
      f1_all =np.concatenate((f1_all,[fold_f1]))  
      bac_all =np.concatenate((bac_all,[fold_bac])) 
      

print()    
print('acc:',acc/5)
print('acc_std:',np.std(acc_all))
print('AUC:',AUC/5)
print('auc_std:',np.std(auc_all))
print('sen:',rec/5)
print('sen_std:',np.std(sen_all))
print('spe:',spe/5)
print('spe_std:',np.std(spe_all))
print('ppv:',pre/5)
print('ppv_std:',np.std(ppv_all))
print('npv:',npv/5)
print('npv_std:',np.std(npv_all))
print('f1:',f1/5)
print('f1_std:',np.std(f1_all))
print('bac:',bac/5)
print('bac_std:',np.std(bac_all))

