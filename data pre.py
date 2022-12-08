#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#load the raw data
#data = pd.read_csv('data_by_year.csv')
data = pd.read_csv('full_music_data.csv')


# In[2]:


data['tempo_new']=0
data['loudness_new']=0
data['duration_ms_new']=0

data['tempo_new']=(data['tempo']-data['tempo'].min())/(data['tempo'].max()-data['tempo'].min())
data['loudness_new']=(data['loudness']-data['loudness'].min())/(data['loudness'].max()-data['loudness'].min())
data['duration_ms_new']=(data['duration_ms']-data['duration_ms'].min())/(data['duration_ms'].max()-data['duration_ms'].min())

data1= data.drop(['artist_names',"artists_id","tempo",'loudness','duration_ms','popularity','release_date','song_title (censored)','year'], 1)
data1
#data.to_csv('new_data_by_year.csv')


# In[3]:



data2=data1-data1.mean()


# In[4]:


import numpy as np; import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
df = data2[0:15]
pd.DataFrame(cosine_similarity(df))


# In[5]:


import numpy as np; import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = data1[0:10]
df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)

pd.DataFrame(df.corr('spearman'))
#pd.DataFrame(df.corr('kendall'))


# In[6]:



df = data1[0:20]
pd.DataFrame(cosine_similarity(df))


# In[7]:


import numpy as np
from scipy.spatial.distance import pdist
import random

x=data1[0:1]
list4=[]
for i in range(15):
    t=random.randint(0,90000)
    list4.append(t)
    
list1=[]
for i in list4:
    list3=[]
    x=data1[i:i+1]
    for j in list4:
        y=data1[j:j+1]
        X=np.vstack([x,y])
        list3.append(pdist(X,'minkowski',p=2)[0])
    list1.append(list3)


#list2=1-(pd.Series(list)-pd.Series(list).min())/(pd.Series(list).max()-pd.Series(list).min())
a=pd.DataFrame(list1)
a


# In[17]:


df = data1.iloc[[134,5552,3087,4015,365,6,388,1,0],:]
pd.DataFrame(cosine_similarity(df))


# In[31]:


data['song_title (censored)'][list4]


# In[41]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = data['song_title (censored)'][list4]
y =data['song_title (censored)'][list4]
sim = a

fig, ax = plt.subplots()
im = ax.imshow(sim)

# We want to show all ticks...
ax.set_xticks(np.arange(len(x)))
ax.set_yticks(np.arange(len(y)))
# ... and label them with the respective list entries
ax.set_xticklabels(x)
ax.set_yticklabels(y)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

# Loop over data dimensions and create text annotations.
for i in range(len(x)):
    for j in range(len(y)):
        text = ax.text(sim[i, j],j,i,
                       ha="center", va="center", color="w")
ax.set_title("Similarity")
fig.tight_layout()
plt.show()


# In[6]:


pd.DataFrame(df.corr('kendal'))


# In[58]:


import numpy as np
y=data1[0:1]
x=data1[1:2]

from scipy.spatial.distance import pdist
X=np.vstack([x,y])
d2=pdist(X,'mahalanobis')

d2


# In[ ]:





# In[60]:


x=data1[8:9]
y=data1[1:2]
from scipy.spatial.distance import pdist
X=np.vstack([x,y])
d2=1-pdist(X,'cosine')

d2


# In[ ]:




