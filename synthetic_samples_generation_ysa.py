import imblearn
from imblearn.over_sampling import SMOTE,ADASYN,BorderlineSMOTE,RandomOverSampler
from sklearn.datasets import  make_classification
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from numpy import where 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
from imblearn.pipeline import Pipeline
from numpy import mean
from scipy.io import arff
import pandas as pd
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import initializers
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

def ortalamaBul(vektor):
    veriAdedi = len(vektor)
    if veriAdedi <= 1:
        return vektor
    else:
        return sum(vektor) / veriAdedi
    
def standartSapmaBul(vektor):
    sd = 0.0 # standart sapma
    veriAdedi = len(vektor)
    if veriAdedi <= 1:
        return 0.0
    else:
        for _ in vektor:
            sd += (float(_) - ortalamaBul(vektor)) ** 2
        sd = (sd / float(veriAdedi)) ** 0.5
        return sd   
def standartSapmaBulv2(vektor,deg):
    sd = 0.0 # standart sapma
    veriAdedi = len(vektor)
    if veriAdedi <= 1:
        return 0.0
    else:
        for _ in vektor:
            sd += (float(_) - deg) ** 2
        sd = (sd / float(veriAdedi)) ** 0.5
        return sd   
    
def varyansBul(vektor):
     return standartSapmaBul(vektor) ** 2
 
def varyansBulv2(vektor,deg):
     return standartSapmaBulv2(vektor,deg) ** 2
 
def windrawlossstarting(acc_smote,acc_bordersmote,acc_adasyn,dataset_name):
    smotevsstart={dataset_name+'_win_smote':0,dataset_name+'_draw':0,dataset_name+'_win_start':0}#win:bagging    
    bordervsstart={dataset_name+'_win_border':0,dataset_name+'_draw':0,dataset_name+'_win_start':0}  #win:bagging
    adasynvsstart={dataset_name+'_win_adasyn':0,dataset_name+'_draw':0,dataset_name+'_win_start':0}  #win:rf

    for c in range(len(acc_smote)-1):    
            if acc_smote[c+1]>acc_smote[0]:
                smotevsstart[dataset_name+'_win_smote']+=1
            elif acc_smote[c+1]==acc_smote[0]:
                smotevsstart[dataset_name+'_draw']+=1
            else:
                smotevsstart[dataset_name+'_win_start']+=1
            
            if acc_bordersmote[c+1]>acc_bordersmote[0]:
                bordervsstart[dataset_name+'_win_border']+=1
            elif acc_bordersmote[c+1]==acc_bordersmote[0]:
                bordervsstart[dataset_name+'_draw']+=1
            else:
                bordervsstart[dataset_name+'_win_start']+=1
                
            
            if acc_adasyn[c+1]>acc_adasyn[0]:
                adasynvsstart[dataset_name+'_win_adasyn']+=1
            elif acc_adasyn[c+1]==acc_adasyn[0]:
                adasynvsstart[dataset_name+'_draw']+=1
            else:
                adasynvsstart[dataset_name+'_win_start']+=1
    return smotevsstart,bordervsstart,adasynvsstart

def windrawlossbtwalgorithm(acc_smote,acc_bordersmote,acc_adasyn,dataset_name):
    smotevsborder={dataset_name+'_win_smote':0,dataset_name+'_draw':0,dataset_name+'_win_border':0}#win:bagging    
    smotevsadasyn={dataset_name+'_win_smote':0,dataset_name+'_draw':0,dataset_name+'_win_adasyn':0}  #win:bagging
    adasynvsborder={dataset_name+'_win_adasyn':0,dataset_name+'_draw':0,dataset_name+'_win_border':0}  #win:rf

    for c in range(len(acc_smote)-1):
        if acc_smote[c+1]>acc_bordersmote[c+1]:
            smotevsborder[dataset_name+'_win_smote']+=1
        elif acc_smote[c+1]==acc_bordersmote[c+1]:
            smotevsborder[dataset_name+'_draw']+=1
        else:
            smotevsborder[dataset_name+'_win_border']+=1
            
        if acc_smote[c+1]>acc_adasyn[c+1]:
            smotevsadasyn[dataset_name+'_win_smote']+=1
        elif acc_smote[c+1]==acc_adasyn[c+1]:
            smotevsadasyn[dataset_name+'_draw']+=1
        else:
            smotevsadasyn[dataset_name+'_win_adasyn']+=1
            
        if acc_adasyn[c+1]>acc_bordersmote[c+1]:
            adasynvsborder[dataset_name+'_win_adasyn']+=1
        elif acc_adasyn[c+1]==acc_bordersmote[c+1]:
            adasynvsborder[dataset_name+'_draw']+=1
        else:
            adasynvsborder[dataset_name+'_win_border']+=1
    return smotevsborder,smotevsadasyn,adasynvsborder
    

alldataset = []
std_dev = []
btw_alg = []
btw_start = []

# arr=['zoo','waveform','vowel','vote','vehicle','splice','soybean']#,'hepatitis']
# inp={'zoo':16,'waveform':40,'vowel':11,'vote':16,'vehicle':18,'splice':287,'soybean':83}
# outp={'zoo':4,'waveform':3,'vowel':11,'vote':2,'vehicle':4,'splice':3,'soybean':18}

arr=['primary-tumor','abalone']
outp={'primary-tumor':11,'abalone':19}
inp={'primary-tumor':23,'abalone':10}



#dataset_name = "soybean"

for dataset_name in arr:
    data = arff.loadarff(dataset_name+'.arff')
    #data = arff.loadarff('abalone.arff')
    df = pd.DataFrame(data[0])
    
    
    X=df.drop(['class'],axis=1)
    y=df.loc[:,'class']
    
    X,y=shuffle(X,y)
    
    
    counter = Counter(y)
    print(counter)
    
    kf = KFold(n_splits=5,random_state=None)
    kf.get_n_splits(X)
    
    # for train_index, test_index in kf.split(X):
    #         X_train2, X_test2 = X.iloc[train_index], X.iloc[test_index]
    #         y_train2, y_test2 = y.iloc[train_index], y.iloc[test_index]
    #         print(Counter(y_test2))
    
    acc_smote=[]
    data_size_smote=[]
    acc_adasyn=[]
    data_size_adasyn=[]
    acc_bordersmote=[]
    data_size_bordersmote=[]
    
    acc=[]
    
    ohe = OneHotEncoder()
    y=pd.DataFrame(y)
    #Herhangi bir yapay örnek üretimi olmadan
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        
        y_train = ohe.fit_transform(y_train).toarray()
        y_test = ohe.fit_transform(y_test).toarray()
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        model = Sequential()
        model.add(Dense(20,kernel_initializer='random_normal', input_dim=inp[''+dataset_name+''], activation='relu'))
        model.add(Dense(150,kernel_initializer='random_normal', activation='relu'))
        model.add(Dense(75,kernel_initializer='random_normal', activation='relu'))
        model.add(Dense(outp[''+dataset_name+''],kernel_initializer='random_normal', activation='softmax'))
        # compile the keras model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=100, batch_size=36)
    
        _,accuracy = model.evaluate(X_test, y_test)
        acc.append(accuracy)
    
    acc_smote.append(mean(acc))
    data_size_smote.append(len(X))
    
    acc_adasyn.append(mean(acc))
    data_size_adasyn.append(len(X))
    
    acc_bordersmote.append(mean(acc))
    data_size_bordersmote.append(len(X))
    
    #SMOTE algoritmasının kullanarak yapay veri üretimi yapılarak
    yapay_sample=[]
    max_num=max(counter.values())
    class_num=len(counter)
    
    winsmote=0
    drawsmote=0
    winborder=0
    drawborder=0
    winadasyn=0
    drawadasyn=0
    inc = (max_num)/2
    for i in range(5):
        if i==0:
            yapay_sample.append(2*max_num)
        else:
            yapay_sample.append(int(2*max_num+i*inc))
        
        
    
    
    # counter_tr = Counter(y_train)
    # print(counter_tr)
    
    threshold=0
    for ys in yapay_sample:
        acc1=[]
        acc2=[]
        acc3=[]
        
        class_dist={}
        for i in(range(len(counter)+1)):
            if counter[i]!=0:
                dct={i:ys}
                class_dist = {**class_dist, **dct}    
                    
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            
            if threshold==0:#(len(yapay_sample)/2):
                X_embedded = TSNE(n_components=2).fit_transform(X_train)
                
                for label, _ in counter.items():
                    row_ix = where(y_train == int(label))[0]
                    pyplot.scatter(X_embedded[row_ix, 0], X_embedded[row_ix, 1], label=str(int(label)))
                pyplot.title("Original Dataset - Dataset:"+dataset_name)
                pyplot.legend()
                pyplot.show()
                
            sm = SMOTE(sampling_strategy=class_dist)
            X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
            X_train_res, y_train_res=shuffle(X_train_res, y_train_res)
        
            y_train_resv2 = ohe.fit_transform(y_train_res).toarray()
            y_testv2 = ohe.fit_transform(y_test).toarray()
            y_train_resv2 = pd.DataFrame(y_train_resv2)
            y_testv2 = pd.DataFrame(y_testv2)
            model = Sequential()
            model.add(Dense(20,kernel_initializer='random_normal', input_dim=inp[''+dataset_name+''], activation='relu'))
            model.add(Dense(75,kernel_initializer='random_normal', activation='relu'))
            model.add(Dense(outp[''+dataset_name+''],kernel_initializer='random_normal', activation='softmax'))
            # compile the keras model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train_res, y_train_resv2, epochs=100, batch_size=36)
        
            _,accuracy = model.evaluate(X_test, y_testv2)
            acc1.append(accuracy)   
        
            
            if threshold==0:#(len(yapay_sample)/2):
                X_embedded = TSNE(n_components=2).fit_transform(X_train_res)
                for label, _ in counter.items():
                    row_ix = where(y_train == int(label))[0]
                    pyplot.scatter(X_embedded[row_ix, 0], X_embedded[row_ix, 1], label=str(int(label)))
                pyplot.title("Sythentitic Data with Smote - Dataset:"+dataset_name)
                pyplot.legend()
                pyplot.show()
                
            
            smborder=BorderlineSMOTE(sampling_strategy=class_dist)
            X_train_res, y_train_res = smborder.fit_sample(X_train, y_train)
            X_train_res, y_train_res=shuffle(X_train_res, y_train_res)
            
            y_train_resv2 = ohe.fit_transform(y_train_res).toarray()
            y_testv2 = ohe.fit_transform(y_test).toarray()
            y_train_resv2 = pd.DataFrame(y_train_resv2)
            y_testv2 = pd.DataFrame(y_testv2)
            model = Sequential()
            model.add(Dense(20,kernel_initializer='random_normal', input_dim=inp[''+dataset_name+''], activation='relu'))
            model.add(Dense(75,kernel_initializer='random_normal', activation='relu'))
            model.add(Dense(outp[''+dataset_name+''],kernel_initializer='random_normal', activation='softmax'))
            # compile the keras model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train_res, y_train_resv2, epochs=100, batch_size=36)
        
            _,accuracy = model.evaluate(X_test, y_testv2)
            acc2.append(accuracy)       
            
            
            if threshold==0:#(len(yapay_sample)/2):
                X_embedded = TSNE(n_components=2).fit_transform(X_train_res)
                for label, _ in counter.items():
                    row_ix = where(y_train == int(label))[0]
                    pyplot.scatter(X_embedded[row_ix, 0], X_embedded[row_ix, 1], label=str(int(label)))
                pyplot.title("Sythentitic Data with BorderlineSmote - Dataset:"+dataset_name)
                pyplot.legend()
                pyplot.show()
            
            
            random=RandomOverSampler(sampling_strategy=class_dist)
            X_train_res, y_train_res = random.fit_sample(X_train, y_train)
            X_train_res, y_train_res=shuffle(X_train_res, y_train_res)
            
            
            y_train_resv2 = ohe.fit_transform(y_train_res).toarray()
            y_testv2 = ohe.fit_transform(y_test).toarray()
            y_train_resv2 = pd.DataFrame(y_train_resv2)
            y_testv2 = pd.DataFrame(y_testv2)
            model = Sequential()
            model.add(Dense(20,kernel_initializer='random_normal', input_dim=inp[''+dataset_name+''], activation='relu'))
            model.add(Dense(75,kernel_initializer='random_normal', activation='relu'))
            model.add(Dense(outp[''+dataset_name+''],kernel_initializer='random_normal', activation='softmax'))
            # compile the keras model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train_res, y_train_resv2, epochs=100, batch_size=36)
        
            _,accuracy = model.evaluate(X_test, y_testv2)
            acc3.append(accuracy)  
            
            if threshold==0:#(len(yapay_sample)/2):
                X_embedded = TSNE(n_components=2).fit_transform(X_train_res)
                for label, _ in counter.items():
                    row_ix = where(y_train == int(label))[0]
                    pyplot.scatter(X_embedded[row_ix, 0], X_embedded[row_ix, 1], label=str(int(label)))
                pyplot.title("Sythentitic Data with Random - Dataset:"+dataset_name)
                pyplot.legend()
                pyplot.show()
                threshold = threshold + 1
            
        threshold=threshold+1
            
        
        acc_smote.append(mean(acc1))            
        acc_bordersmote.append(mean(acc2))          
        acc_adasyn.append(mean(acc3))

        data_size_smote.append(ys*len(class_dist))
        data_size_bordersmote.append(ys*len(class_dist))
        data_size_adasyn.append(ys*len(class_dist))
        
    
    xx = {dataset_name+"_smote":acc_smote}
    yy = {dataset_name+"_bordersmote":acc_bordersmote}
    zz = {dataset_name+"_adasyn":acc_adasyn}
    alldataset.append(xx)
    alldataset.append(yy)
    alldataset.append(zz)
    
    xx = {dataset_name+"_smote":standartSapmaBul(acc_smote)}
    yy = {dataset_name+"_bordersmote":standartSapmaBul(acc_bordersmote)}
    zz = {dataset_name+"_adasyn":standartSapmaBul(acc_adasyn)}
    std_dev.append(xx)
    std_dev.append(yy)
    std_dev.append(zz)
    
    xx,yy,zz = windrawlossbtwalgorithm(acc_smote,acc_bordersmote,acc_adasyn,dataset_name)
    btw_alg.append(xx)
    btw_alg.append(yy)
    btw_alg.append(zz)
    
    xx,yy,zz=windrawlossstarting(acc_smote,acc_bordersmote,acc_adasyn,dataset_name)
    btw_start.append(xx)
    btw_start.append(yy)
    btw_start.append(zz)
    
    k=plt.plot(data_size_smote,acc_smote,'r-o',data_size_adasyn,acc_adasyn,'g-o',data_size_bordersmote,acc_bordersmote,'b-o')
    #m=plt.plot(data_size_smote[0],acc_smote[0],'b*',data_size_adasyn[0],acc_adasyn[0],'b*')
    plt.legend([k[0],k[1],k[2]],["SMOTE","Random",'BorderlineSMOTE'])
    plt.title("Dataset Size versus Accuracy Graph - Dataset Name:"+dataset_name)
    plt.xlabel("Dataset Size")
    plt.ylabel("Accuracy")
    plt.show()
    
    
import pickle

with open("C:/Users/ITU/Desktop/Kolektif proje/DataRes/std_devt_last36.txt", "wb") as fp:   #Pickling
    pickle.dump(std_dev, fp)
with open("C:/Users/ITU/Desktop/Kolektif proje/DataRes/btw_start_last36.txt", "wb") as fp:   #Pickling
    pickle.dump(btw_start, fp)
with open("C:/Users/ITU/Desktop/Kolektif proje/DataRes/btw_alg_last36.txt", "wb") as fp:   #Pickling
    pickle.dump(btw_alg, fp)
with open("C:/Users/ITU/Desktop/Kolektif proje/DataRes/alldataset_last36.txt", "wb") as fp:   #Pickling
    pickle.dump(alldataset, fp)


with open("C:/Users/ITU/Desktop/Kolektif proje/DataRes/std_devt_last28pr.txt", "rb") as fp:   # Unpickling
    std_dev = pickle.load(fp)
with open("C:/Users/ITU/Desktop/Kolektif proje/DataRes/btw_start_last28pr.txt", "rb") as fp:   # Unpickling
    btw_start = pickle.load(fp)
with open("C:/Users/ITU/Desktop/Kolektif proje/DataRes/btw_alg_last28pr.txt", "rb") as fp:   # Unpickling
    btw_alg = pickle.load(fp)
with open("C:/Users/ITU/Desktop/Kolektif proje/DataRes/alldataset_last28pr.txt", "rb") as fp:   # Unpickling
    alldataset = pickle.load(fp)

        
###################################### k en yakın örneğin değişimine göre algoritma başarıları###############################################
acc_smotev2=[]
acc_adasynv2=[]  
acc_bordersmotev2=[]

k_val=[1,5,9]
for k in k_val:
    tmp1=[]
    tmp2=[]
    tmp3=[]
    for ys in yapay_sample:
        acc1=[]
        acc2=[]
        acc3=[]
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            class_dist={}
            for i in(range(len(counter)+1)):
                if counter[i]!=0:
                    dct={i:ys}
                    class_dist = {**class_dist, **dct} 
            
            
            sm = SMOTE(sampling_strategy=class_dist,k_neighbors=k)
            X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
        
            model = DecisionTreeClassifier()
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            acc1.append(accuracy_score(y_test, y_pred))
            
            adasyn = ADASYN(sampling_strategy=class_dist,n_neighbors=k)
            X_train_res, y_train_res = adasyn.fit_sample(X_train, y_train.ravel())
        
            model = DecisionTreeClassifier()
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            acc2.append(accuracy_score(y_test, y_pred))
            
            bordersm = BorderlineSMOTE(sampling_strategy=class_dist,k_neighbors=k)
            X_train_res, y_train_res = bordersm.fit_sample(X_train, y_train.ravel())
        
            model = DecisionTreeClassifier()
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            acc3.append(accuracy_score(y_test, y_pred))      
            
        tmp1.append(mean(acc1))
        tmp2.append(mean(acc2))
        tmp3.append(mean(acc2))
        
    acc_smotev2.append(tmp1)
    acc_adasynv2.append(tmp2)  
    acc_bordersmotev2.append(tmp3)

del data_size_smote[0]
del data_size_adasyn[0]
del data_size_bordersmote[0]

plt.plot(data_size_smote,acc_smotev2[0],data_size_smote,acc_smotev2[1],data_size_smote,acc_smotev2[2])
plt.legend(["k=1","k=5","k=9"])
plt.title("Dataset Size versus Accuracy Graph with Different k Neighbor-SMOTE")
plt.xlabel("Dataset Size")
plt.ylabel("Accuracy")
plt.show()


print("k=1 variance:%lf" % standartSapmaBul(acc_smotev2[0]))
print("k=5 variance:%lf" % standartSapmaBul(acc_smotev2[1]))
print("k=9 variance:%lf" % standartSapmaBul(acc_smotev2[2]))
#print("k=11 variance:%lf" % standartSapmaBul(acc_smotev2[3]))


plt.plot(data_size_adasyn,acc_adasynv2[0],data_size_adasyn,acc_adasynv2[1],data_size_adasyn,acc_adasynv2[2])
plt.legend(["k=1","k=5","k=9"])
plt.title("Dataset Size versus Accuracy Graph with Different k Neighbor-ADASYN")
plt.xlabel("Dataset Size")
plt.ylabel("Accuracy")
plt.show()

print("k=1 variance:%lf" % standartSapmaBul(acc_adasynv2[0]))
print("k=5 variance:%lf" % standartSapmaBul(acc_adasynv2[1]))
print("k=9 variance:%lf" % standartSapmaBul(acc_adasynv2[2]))
#print("k=11 variance:%lf" % standartSapmaBul(acc_adasynv2[3]))


plt.plot(data_size_smote,acc_bordersmotev2[0],data_size_smote,acc_bordersmotev2[1],data_size_smote,acc_bordersmotev2[2])
plt.legend(["k=1","k=5","k=9"])
plt.title("Dataset Size versus Accuracy Graph with Different k Neighbor-BorderlineSmote")
plt.xlabel("Dataset Size")
plt.ylabel("Accuracy")
plt.show()

print("k=1 variance:%lf" % standartSapmaBul(acc_bordersmotev2[0]))
print("k=3 variance:%lf" % standartSapmaBul(acc_bordersmotev2[1]))
print("k=7 variance:%lf" % standartSapmaBul(acc_bordersmotev2[2]))
#print("k=11 variance:%lf" % standartSapmaBul(acc_bordersmotev2[3]))

