#!/usr/bin/env python
# coding: utf-8

# In[100]:


import gzip
import numpy as np
import sys
import matplotlib.pyplot as plt

def training_images():
    with gzip.open('train-images-idx3-ubyte.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        column_count = int.from_bytes(f.read(4), 'big')
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)            .reshape((image_count, row_count*column_count))
        return images


def training_labels():
    with gzip.open('train-labels-idx1-ubyte.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        label_count = int.from_bytes(f.read(4), 'big')
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels
def test_labels():
    with gzip.open('t10k-labels-idx1-ubyte.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        label_count = int.from_bytes(f.read(4), 'big')
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels
def test_images():
    with gzip.open('t10k-images-idx3-ubyte.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        column_count = int.from_bytes(f.read(4), 'big')
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)            .reshape((image_count, row_count* column_count))
        return images


# In[101]:


train_images=training_images()
train_labels=training_labels()
test_images=test_images()
test_labels=test_labels()
print(train_images.shape,train_labels.shape,test_images.shape,test_labels.shape)
train_images.squeeze()
train_images.shape


# In[102]:


allclasses=[];
for i in range(10):
    allclasses.append([]);
for i in range(60000):
    allclasses[train_labels[i]].append(train_images[i])
for i in range(10):
    allclasses[i]=np.array(allclasses[i]);


# In[129]:


for i in range(10):
    data = allclasses[i].reshape(len(allclasses[i]), 28, 28)
    print("CLASS ", i)
    for j in range(5):
        image = np.asarray(data[j]).squeeze()
        plt.imshow(image,cmap='gray')
        plt.show()


# In[103]:


allclasses=np.array(allclasses,dtype=object)
allclasses.shape


# In[104]:


sum=0
for i in range(10):
    sum+=len(allclasses[i])
    print(len(allclasses[i]))
print(sum)
np.concatenate(allclasses)
print(allclasses[0][1].shape)


# In[130]:


allclassesmean=[]
allclassescov=[]

for i in range(10):
    allclassesmean.append([]);
    allclassescov.append([]);
for i in range(10):
    allclassesmean[i]=np.mean(allclasses[i],axis=0)
    xicentered=allclasses[i]-allclassesmean[i]
    xicentered=np.array(xicentered)
    xixi=np.dot(xicentered.T,xicentered)
    allclassescov[i]=xixi/len(allclasses[i])

    
allclassesmean=np.array(allclassesmean)
allclassescov=np.array(allclassescov)

allclassescov.shape


# In[131]:


allclassesscatter=[]
allclassescentralised=[]
for i in range(10):
    allclassesscatter.append([]);
    allclassescentralised.append([]);
for i in range(10):
    for j in range(len(allclasses[i])):
        allclassescentralised[i].append(allclasses[i][j]-allclassesmean[i]);

for i in range(10):
    allclassescentralised[i]=np.array(allclassescentralised[i])


# In[107]:


allclassescentralised=np.array(allclassescentralised,dtype=object)


# In[108]:


allclassescentralised[1].shape


# In[109]:


for i in range(10):
    allclassesscatter[i]=np.dot(allclassescentralised[i].T,allclassescentralised[i])


# In[110]:


for i in range(10):
    allclassesscatter[i]=np.array(allclassesscatter[i])
allclassesscatter=np.array(allclassesscatter)


# In[111]:


allclassesscatter.shape
allclassesscatter[7]


# In[112]:


Sw=np.zeros((784,784));
for i in range(10):
    Sw+=allclassesscatter[i];


X=train_images     
ug=np.mean(X,axis=0)
print(ug.shape)
Xcentralised=X-ug
print(Xcentralised.shape)
St=np.dot(Xcentralised.T,Xcentralised)
print(St.shape)


# In[113]:


Sb=St-Sw
print(Sb.shape)


# In[114]:


Swinv=np.linalg.pinv(Sw)


# In[115]:


E=np.dot(Swinv,Sb)


# In[116]:


E.shape


# In[117]:


evalues, evectors = np.linalg.eig(E)


# In[118]:


evectors=evectors.T
print(evalues[45],evalues[53],evalues[54])


# In[119]:


idx=np.argsort(abs(evalues))[::-1]

evalues=evalues[idx]
evectors=evectors[idx]
evalues=evalues[:9]
W=evectors[:9]
W=W.T


# In[120]:


print(W.shape,X.shape)


# In[121]:


Y=np.dot(W.T,X.T)


# In[122]:


Y.shape
Yallclasses=[];
Y=Y.T
for i in range(10):
    Yallclasses.append([]);
for i in range(60000):
    Yallclasses[train_labels[i]].append(Y[i])
for i in range(10):
    Yallclasses[i]=np.array(Yallclasses[i]);
Yallclasses=np.array(Yallclasses,dtype=object)
Yallclassesmean=[]
Yallclassescov=[]

for i in range(10):
    Yallclassesmean.append([]);
    Yallclassescov.append([]);
for i in range(10):
    Yallclassesmean[i]=np.mean(Yallclasses[i],axis=0)
    xicentered=Yallclasses[i]-Yallclassesmean[i]
    xicentered=np.array(xicentered)
    xixi=np.dot(xicentered.T,xicentered)
    Yallclassescov[i]=xixi/len(Yallclasses[i])
    print(xicentered.shape)
    
Yallclassesmean=np.array(Yallclassesmean)
Yallclassescov=np.array(Yallclassescov)

Yallclassescov.shape


# In[123]:


def g(x,x_cov,x_mean):
    x_cov_inv= np.linalg.inv(x_cov)
    t1=np.dot(np.dot(x.T,(-1/2)*x_cov_inv),x)
    t2=np.dot(np.dot(x_cov_inv,x_mean).T,x)
    t3= np.dot(np.dot(((-1/2)*(x_mean.T)),x_cov_inv),x_mean) - 1/2* np.log(np.linalg.det(x_cov_inv)) + np.log(1/2)
    return t1+t2+t3


# In[124]:


Y_test=np.dot(W.T,test_images.T)
Y_test.shape


# In[125]:



Y_pred=[]
v=[]
maxi=-1000000;
indexm=-1;
for i in range(10000):
    for j in range(10):
        tem=g(Y_test[:,i:i+1],Yallclassescov[j],Yallclassesmean[j])
        if(tem>maxi):
            maxi=tem
            indexm=j
    Y_pred.append(indexm)
    maxi=-1000000;
    indexm=-1;
   

mismatched=0;
for i in range(10000):
#     print(Y_pred[i],test_labels[i])
    if Y_pred[i]!=test_labels[i]:
        mismatched+=1
print(mismatched)


# In[126]:


accuracy=(10000-mismatched)/10000
print(accuracy)


# In[ ]:




