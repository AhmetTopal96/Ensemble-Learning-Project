
import matplotlib.pyplot as plt

import pandas as pd 


import pickle


with open("C:/Users/ITU/Desktop/Kolektif proje/DataRes/std_devt_last36.txt", "rb") as fp:   # Unpickling
    std_dev = pickle.load(fp)
with open("C:/Users/ITU/Desktop/Kolektif proje/DataRes/btw_start_last36.txt", "rb") as fp:   # Unpickling
    btw_start = pickle.load(fp)
with open("C:/Users/ITU/Desktop/Kolektif proje/DataRes/btw_alg_last36.txt", "rb") as fp:   # Unpickling
    btw_alg = pickle.load(fp)
with open("C:/Users/ITU/Desktop/Kolektif proje/DataRes/alldataset_last36.txt", "rb") as fp:   # Unpickling
    alldataset = pickle.load(fp)

tmp=alldataset
dataname=[
'zoo',
'waveform',
'vowel',
'vote',
'vehicle',
'splice',
'soybean',
'sonar',
'sick',
'segment',
'ringnorm',
'mushroom',
'lymph',
'letter',
'labor',
'kr-vs-kp',
'iris',
'ionosphere',
'hypothyroid',
'hepatitis',
'heart-statlog',
'glass',
'diabetes',
'd159',
'credit-g',
'credit-a',
'colic',
'col10',
'breast-w',
'breast-cancer',
'balance-scale',
'autos',
'audiology',
'anneal',
'primary-tumor',
'abalone',
'spam message',
'Tweet_RealorNot']

accval=[]
i=0
graph_name=[]
while i<len(alldataset):
    for key,value in alldataset[i].items():
        graph_name.append(key)
        accval.append(value)
    i=i+1
    
onlysynt_acc=[]
orig_acc=[]
for i in range(len(accval)):
    orig_acc.append(accval[i][0])
    onlysynt_acc.append(accval[i][1:])
    
max_synth_acc_smote=[]
max_synth_acc_border=[]
max_synth_acc_random=[]
orig_acc_plot=[]
for i in range(len(onlysynt_acc)):
    if i%3==0:
        orig_acc_plot.append(orig_acc[i])
        max_synth_acc_smote.append(max(onlysynt_acc[i]))
    if (i%3==1):
        max_synth_acc_border.append(max(onlysynt_acc[i]))
    if (i%3==2):
        max_synth_acc_random.append(max(onlysynt_acc[i]))
    
df = pd.DataFrame({'original':orig_acc_plot,
                   'smote':max_synth_acc_smote,
                   'BorderSmote':max_synth_acc_border,
                   'Random':max_synth_acc_random},
                   index=dataname)

smotevsborder=[]
smotevsrandom=[]
randomvsborder=[]

for i in range(len(btw_alg)):
    if i%3==0:
        smotevsborder.append(btw_alg[i])
    if (i%3==1):
        smotevsrandom.append(btw_alg[i])
    if (i%3==2):
        randomvsborder.append(btw_alg[i])
        

smotevsstart=[]
bordervsstart=[]
randomvsstart=[]

for i in range(len(btw_alg)):
    if i%3==0:
        smotevsstart.append(btw_start[i])
    if (i%3==1):
        bordervsstart.append(btw_start[i])
    if (i%3==2):
        randomvsstart.append(btw_start[i])
    
df = pd.DataFrame({'original':orig_acc_plot,
                   'smote':max_synth_acc_smote,
                   'BorderSmote':max_synth_acc_border,
                   'Random':max_synth_acc_random},
                   index=dataname)

import numpy as np
ori = df.original
smo = df.smote
bor = df.BorderSmote
ran = df.Random

ind = np.arange(df.shape[0])
width = 0.05



# def autolabel(bars):
#     # attach some text labels
#     for bar in bars:
#         width = bar.get_width()
#         ax.text(width*0.95, bar.get_y() + bar.get_height()/2,
#                 '%d' % int(width),
#                 ha='right', va='center')

# make the plots
fig, ax = plt.subplots()

original = ax.barh(ind, ori, width) # plot a vals
smote = ax.barh(ind + width, smo, width,  alpha=0.5)  # plot b vals
BorderSmote = ax.barh(ind + 2*width, bor, width,  alpha=0.5)
Random = ax.barh(ind + 3*width, ran, width, alpha=0.5)

ax.set_yticks(ind + 1.5*width)  # position axis ticks
ax.set_yticklabels(df.index)  # set them to the names
ax.legend((original[0], smote[0],BorderSmote[0],Random[0]), ['Original', 'Smote','BorderSmote','Random'], loc='lower left')
plt.title("Original and Maximum Synthetic Accuracy")
# autolabel(original)
# autolabel(smote)
# autolabel(BorderSmote)
# autolabel(Random)
plt.ylim(top=24,bottom=13)
plt.axis([0,1,35.7,37.5])
plt.show()

rank=[]
for i in range(len(dataname)):
    smo=1
    bord=1
    orig=1
    rand=1
    
    a=list(smotevsstart[i].values())
    if a[0]>a[2]:
        orig+=1
    elif a[0]<a[2]:
        smo+=1
        
    a=list(bordervsstart[i].values())
    if a[0]>a[2]:
        orig+=1
    elif a[0]<a[2]:
        bord+=1
  
    a=list(randomvsstart[i].values())
    if a[0]>a[2]:
        orig+=1
    elif a[0]<a[2]:
        rand+=1

    a=list(smotevsborder[i].values())
    if a[0]>a[2]:
        bord+=1
    elif a[0]<a[2]:
        smo+=1

    a=list(smotevsrandom[i].values())
    if a[0]>a[2]:
        rand+=1
    elif a[0]<a[2]:
        smo+=1
    
    a=list(randomvsborder[i].values())
    if a[0]>a[2]:
        bord+=1
    elif a[0]<a[2]:
        rand+=1
    cond = {'data':dataname[i],'orig_rank':orig,'smote_rank':smo,'random_rank':rand,'border_rank':bord}
    rank.append(cond)
    
balance=['credit-a','d159','iris','kr-vs-kp','letter','mushroom','ringnorm','segment','sonar','vehicle','vowel','waveform']
partial_balance=['heart-statlog','lymph']
partial_not_balance=['audiology','autos','breast-cancer','breast-w','colic','credit-g','diabetes','ionosphere','labor','splice','vote','Tweet_RealorNot']
not_balance=['abalone','anneal','balance-sclae','col10','glass','hepatitis','hypothyroid','primary-tumor','sick','soybean','zoo','spam_message']

smo=0
orig=0
bord=0
rand=0
for dset in partial_not_balance:
    for k in range(len(rank)):
        if rank[k]['data']==dset:
            smo+=rank[k]['smote_rank']
            orig+=rank[k]['orig_rank']
            bord+=rank[k]['border_rank']
            rand+=rank[k]['random_rank']


ort_smo=smo/len(partial_not_balance)
ort_orig=orig/len(partial_not_balance)
ort_bord=bord/len(partial_not_balance)
ort_rand=rand/len(partial_not_balance)



temp=[]
for i in range(len(std_dev)):
    if i%3==0:
        k=list(std_dev[i].values())
        tmp={'data':dataname[int(i/3)],'std_smote':k[0]}
        temp.append(tmp)
    if i%3==1:
        k=list(std_dev[i].values())
        tmp={'data':dataname[int(i/3)],'std_border':k[0]}
        temp.append(tmp)
    if i%3==2:
        k=list(std_dev[i].values())
        tmp={'data':dataname[int(i/3)],'std_random':k[0]}
        temp.append(tmp)


stdsmote=0
stdborder=0
stdrandom=0

for dset in partial_not_balance:
    for i in range(len(temp)):
        if dset == temp[i]['data']:
            if i%3==0:
                stdsmote+=temp[i]['std_smote']
            if (i%3==1):
                stdborder+=temp[i]['std_border']
            if (i%3==2):
                stdrandom+=temp[i]['std_random']

stdsmote=stdsmote/len(partial_not_balance)
stdborder=stdborder/len(partial_not_balance)
stdrandom=stdrandom/len(partial_not_balance)
