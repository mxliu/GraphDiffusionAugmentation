import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import scipy.io
import numpy as np
import torch
from torch.nn import functional as F


       

class MDD(object):
     def read_data(self):
    
        site20 = scipy.io.loadmat(':\SITE20.mat')    
        bold20 =site20['AAL']              
        A20 =bold20[0]
        pc_l=[]
        
        y_list=[]
             
        for i in range(len(A20)): 
            pc= np.corrcoef(A20[i].T)
            pc = np.nan_to_num(pc)
            pc=abs(pc)         
            pc_l.append(pc)
            
            #constract y for global feature
            y_0 = torch.zeros([0]).float()
            y_np = y_0.numpy()
            y_list.append(y_np)
                   
        X =np.array(pc_l) 
        E=torch.from_numpy(X)
        E=torch.where(E >= 0.3, torch.tensor(1.0), torch.tensor(0.0)).long()
        E = F.one_hot(E, num_classes=2)
       
      
        y=np.array(y_list) 
        
        X=torch.from_numpy(X)
          
        y=torch.from_numpy(y)
        
        return  X,E,y
    
    


dataset=MDD()
X1,E1,y=dataset.read_data()
