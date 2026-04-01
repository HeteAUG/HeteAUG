import torch
from torch import nn
from torch.nn import init
import numpy as np
import pandas as pd
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import LabelEncoder,OneHotEncoder, MaxAbsScaler


import sys 
sys.path.append("../..") 
sys.path.append('./Kitsune-py-master') 
from Kitsune import *
import numpy as np
import time
import numpy as np
import pandas as pd
import preprocess as p
import trainer as t
import time
import sys
import pandas as pd
import numpy as np
from torch import nn,optim
#import pyhash 
import gensim
import multiprocessing as mp
from joblib import Parallel, delayed
import concurrent.futures
from pprint import pprint
import random
import mpld3 as mp
import re
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import manifold
from sklearn.decomposition import PCA, TruncatedSVD
import csv
import time
import joblib
from mpl_toolkits.mplot3d import Axes3D
import sklearn as sk
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import torch as th
from torch.autograd import Variable as V
from torch import nn,optim
from tqdm import tqdm_notebook as tqdm
import numpy as np
import random
import copy
#from model import Skipgram

from sko.GA import GA
import numpy as np
import os

minzy=[1487598189.488845000,1,0,0,0,0,0,0,0,0]

maxzy=[1540451017.362977000,65536,7,13,7,12,65535,65535,65535,65535]

precisionzy=[0.000001,1,1,1,1,1,1,1,1,1]


import torch
from torch import nn
from torch.nn import init
import numpy as np
import pandas as pd
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt

D2 = pd.read_csv("./data2/data2guolv.tsv",header=None)

D3 = pd.read_csv("./data2/data3guolv.tsv",header=None)

D4 = pd.read_csv("./data2/data4guolv.tsv",header=None)

D5 = pd.read_csv("./data2/data5guolv.tsv",header=None)


def dist(a, b):
    return np.sqrt(sum((a - b) ** 2))


x_train_aug = np.load("./data/our-feature0413.npy")

print("x_train_aug",x_train_aug.shape)

dff = pd.read_csv("./data/sslbenign.tsv",sep='\t')



def fitness0(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9):
    
    x0=float(x0)
    x1=int(float(x1))
    
    x2=int(float(x2))
    x3=int(float(x3))
    x4=int(float(x4))
    x5=int(float(x5))
    
    x6=int(float(x6))
    x7=int(float(x7))
    x8=int(float(x8))
    x9=int(float(x9))
    
    
    
    if x6==0 or x7==0:
        D6='NaN'
        D7='NaN'
        if x8==0:
            D8=x8+1
        else:
            D8=x8
            
        if x9==0:
            D9=x9+1
        else:
            D9=x9
    else:
        D8='NaN'
        D9='NaN'
        D6=x6
        D7=x7
        
    D55='192.168.3.11'
     
    D33=D3[0][x3]
    if D2[0][x2]==D3[0][x3]:
        D33=D3[0][(x3+1)%14]
    
    
    D44=D4[0][x4]
    
    if D4[0][x4]==D55:
        D44=D4[0][(x4+1)%8]
        
        
        
    new_data=pd.DataFrame({'frame.time_epoch':x0,'frame.len':x1,'eth.src':D2[0][x2], \
    'eth.dst':D33,'ip.src':D44,'ip.dst':D55,'tcp.srcport':D6, \
     'tcp.dstport':D7,	'udp.srcport':D8,	'udp.dstport':D9,	'icmp.type':'NaN',	'icmp.code':'NaN',
     'arp.opcode':'NaN',	'arp.src.hw_mac':'NaN',	'arp.src.proto_ipv4':'NaN',	'arp.dst.hw_mac':'NaN',	'arp.dst.proto_ipv4':'NaN',	'ipv6.src':'NaN',
     'ipv6.dst':'NaN'},index=[100])

    data_test=pd.concat([dff, new_data], axis=0)

    P_dir='./data/'
    l_namemiraindata_train2=P_dir +'sslbenign-header-1w-jiange-add-5w2-5shengyu-0.tsv'
    data_test.to_csv(l_namemiraindata_train2, index=False,sep='\t')

    train_num=101
    packet_limit = np.Inf #the number of packets to process
    maxAE = 10 #maximum size for any autoencoder in the ensemble layer
    FMgrace = 40000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
    ADgrace = train_num #the number of instances used to train the anomaly detector (ensemble itself)
    alpha=10

    path="./data/sslbenign-header-1w-jiange-add-5w2-5shengyu-0.tsv"
    K = Kitsune(path,packet_limit,maxAE,FMgrace,ADgrace)

    #print("Running Kitsune:")
    Xvectors = []
    i = 0
    #start = time.time()
    # Here we process (train/execute) each individual packet.
    # In this way, each observation is discarded after performing process() method.
    while True:
        i+=1
        if i%100==0 :
            pass
            #print(i)
        if i == train_num+1:
            break
        #rmse = K.proc_next_packet()
        vec = K.FE.get_next_vector()
        Xvectors.append(vec)
    #stop = time.time()
    #print("Complete. Time elapsed: "+ str(stop - start))

    Xvectors=np.array(Xvectors)

    from Nomalizor import Normalizor
    n = Normalizor()
    n.fit(Xvectors)
    XX = n.normalize(Xvectors)
    YY=XX[100]
    
    distancezy=dist(aa,YY)

    os.remove('./data/sslbenign-header-1w-jiange-add-5w2-5shengyu-0.tsv')
   
    return distancezy
    
    
    

def getproblemFE(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,pddzy):
    
    x0=float(x0)
    x1=int(float(x1))
    
    x2=int(float(x2))
    x3=int(float(x3))
    x4=int(float(x4))
    x5=int(float(x5))
    
    x6=int(float(x6))
    x7=int(float(x7))
    x8=int(float(x8))
    x9=int(float(x9))
    
    
    
    if x6==0 or x7==0:
        D6='NaN'
        D7='NaN'
        if x8==0:
            D8=x8+1
        else:
            D8=x8
            
        if x9==0:
            D9=x9+1
        else:
            D9=x9
    else:
        D8='NaN'
        D9='NaN'
        D6=x6
        D7=x7
        
        
    D55='192.168.3.11'
     
    D33=D3[0][x3]
    if D2[0][x2]==D3[0][x3]:
        D33=D3[0][(x3+1)%14]
    
    
    D44=D4[0][x4]
    
    if D4[0][x4]==D55:
        D44=D4[0][(x4+1)%8]
        
        
        
    new_data=pd.DataFrame({'frame.time_epoch':x0,'frame.len':x1,'eth.src':D2[0][x2], \
    'eth.dst':D33,'ip.src':D44,'ip.dst':D55,'tcp.srcport':D6, \
     'tcp.dstport':D7,	'udp.srcport':D8,	'udp.dstport':D9,	'icmp.type':'NaN',	'icmp.code':'NaN',
     'arp.opcode':'NaN',	'arp.src.hw_mac':'NaN',	'arp.src.proto_ipv4':'NaN',	'arp.dst.hw_mac':'NaN',	'arp.dst.proto_ipv4':'NaN',	'ipv6.src':'NaN',
     'ipv6.dst':'NaN'},index=[100])

    
    pddzy = pd.concat([pddzy, new_data], axis=0)
    
    data_test=pd.concat([dff, new_data], axis=0)

    P_dir='./data/'
    l_namemiraindata_train2=P_dir +'sslbenign-header-1w-jiange-add-5w2-5shengyu-0.tsv'
    data_test.to_csv(l_namemiraindata_train2, index=False,sep='\t')

    train_num=101
    packet_limit = np.Inf #the number of packets to process
    maxAE = 10 #maximum size for any autoencoder in the ensemble layer
    FMgrace = 40000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
    ADgrace = train_num #the number of instances used to train the anomaly detector (ensemble itself)
    alpha=10

    path="./data/sslbenign-header-1w-jiange-add-5w2-5shengyu-0.tsv"
    K = Kitsune(path,packet_limit,maxAE,FMgrace,ADgrace)

    #print("Running Kitsune:")
    Xvectors = []
    i = 0
    #start = time.time()
    # Here we process (train/execute) each individual packet.
    # In this way, each observation is discarded after performing process() method.
    while True:
        i+=1
        if i%100==0 :
            pass
            #print(i)
        if i == train_num+1:
            break
        #rmse = K.proc_next_packet()
        vec = K.FE.get_next_vector()
        Xvectors.append(vec)
    #stop = time.time()
    #print("Complete. Time elapsed: "+ str(stop - start))

    Xvectors=np.array(Xvectors)

    from Nomalizor import Normalizor
    n = Normalizor()
    n.fit(Xvectors)
    XX = n.normalize(Xvectors)
    YY=XX[100]
    #print("YY:",YY)     
                
    distancezy=dist(aa,YY)
    os.remove('./data/sslbenign-header-1w-jiange-add-5w2-5shengyu-0.tsv')
    return distancezy,YY,pddzy
    
    
smoteproblem=[]
pddzy = pd.read_csv("./data/sslbenign2.tsv")#sslbenign.tsv is the same as sslbenign2.tsv
for i in range(len(x_train_aug)):
    print("i=",i)
    aa=x_train_aug[i]
    ga1 = GA(func=fitness0, n_dim=10, size_pop=50, max_iter=3, lb=minzy , ub=maxzy, precision=precisionzy)
    bestX,bestY= ga1.run()
    print("bestY",bestY)
    diszy,YYzy,pddzy=getproblemFE(bestX[0],bestX[1],bestX[2],bestX[3],bestX[4],bestX[5],bestX[6],bestX[7],bestX[8],bestX[9],pddzy)
    print("diszy",diszy)
    smoteproblem.append(YYzy)


featestsc = np.array(smoteproblem)
featestsc=torch.squeeze(torch.tensor(featestsc)).numpy()


pddzy=pddzy[100:]
pddzy.to_csv('./data/ssl-problem-space.tsv', header=True,index=False,sep='\t') #Solution in the Problem Space

np.save("./data/ssl-problem-space-fea.npy",featestsc)# After feature extraction
print("featestsc",featestsc.shape)









