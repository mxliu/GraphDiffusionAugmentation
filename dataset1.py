import scipy.io
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from einops import rearrange


class MDD(object):
     def read_data(self):
    
        site1 = scipy.io.loadmat(':\\SITE1.mat')    
        bold1 =site1['AAL']              
        A1 =bold1[0]
        pc_l=[]
             
        for i in range(len(A1)): 
            pc= np.corrcoef(A1[i].T)
            pc = np.nan_to_num(pc)
            pc=abs(pc)         
            pc_l.append(pc)
            
        X =np.array(pc_l) 
        
        
        E=X.copy()
        E=np.where(E >= 0.3, 1,0)
        E= rearrange(E, '(b s) t c -> b s t c', b=148, s=1, t=116, c=116) 

        X= rearrange(X, '(b s) t c -> b s t c', b=148, s=1, t=116, c=116) 
        
        y= site1['lab']
        y= np.squeeze(y)
        
        
        return  X,E,y    
     def __init__(self):
        
         super(MDD,self).__init__()
         X,E,y =self.read_data()
         self.X =torch.from_numpy(X)
         self.E =torch.from_numpy(E)
         self.n_samples =X.shape[0]    
     
     def __len__(self):
         return self.n_samples
     def __getitem__(self, index):
          return self.X[index]   


dataset=MDD()
X,E,y=dataset.read_data()

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=9)
fold = 1
for finetune_idx,test_idx in skf.split(X, y):
    
    finetune_X = X[finetune_idx]
    finetune_E = E[finetune_idx]
    finetune_label = y[finetune_idx]
    
    test_X = X[test_idx]
    test_E = E[test_idx]
    test_label = y[test_idx] 

    filename = './finetune_X_'+str(fold)+'.npy'
    np.save(filename,finetune_X)
    filename = './finetune_E_'+str(fold)+'.npy'
    np.save(filename,finetune_E)
    filename = './finetune_label_'+str(fold)+'.npy'
    np.save(filename,finetune_label)
    
    filename = './test_X_'+str(fold)+'.npy'
    np.save(filename,test_X)
    filename = './test_E_'+str(fold)+'.npy'
    np.save(filename,test_E)
    filename = './test_label_'+str(fold)+'.npy'
    np.save(filename,test_label) 
    
    fold = fold + 1
