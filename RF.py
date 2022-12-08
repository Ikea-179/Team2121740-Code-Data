#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=200, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[3]:





# In[4]:





# In[5]:





# In[1]:


import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=200, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[2]:





# In[37]:


import pandas as pd
import numpy as np
import random
data = pd.read_csv('merged_artists.csv')

data['tempo_new']=0
data['loudness_new']=0
data['duration_ms_new']=0

data['tempo_new']=(data['tempo']-data['tempo'].min())/(data['tempo'].max()-data['tempo'].min())
data['loudness_new']=(data['loudness']-data['loudness'].min())/(data['loudness'].max()-data['loudness'].min())
data['duration_ms_new']=(data['duration_ms']-data['duration_ms'].min())/(data['duration_ms'].max()-data['duration_ms'].min())

data1= data.drop(['Unnamed: 0','artist_name',"artist_id","tempo",'loudness','duration_ms','jllable','count','popularity'], 1)
data1


# In[ ]:





# In[ ]:





# In[48]:


'''
Random Forest--predicttion of comment_num,like_num,repost_num
'''
X = data1.drop(['genre'],axis=1)
y = data1['genre']

#import packages needed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = RandomForestClassifier(n_estimators=500, random_state=10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))
importances = classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    
# Plot the feature importances of the forest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection

get_ipython().run_line_magic('matplotlib', 'inline')

fig= plt.figure()
ax=plt.axes()

map_vir = cm.get_cmap(name='Greens')
n=5
norm = plt.Normalize(importances[indices][:n].min(), importances[indices][:n].max())
norm_y = norm(importances[indices][:n])
color = map_vir(norm_y)

plt.title("Feature Importances for Determining Genre")

plt.bar(range(5), importances[indices][:5],
       color=color, yerr=std[indices][:5], align="center")
plt.xticks(range(5), ['Acousticness','Danceability','Energy','Instrumentalness','Speechiness'])

plt.xlim([-1, 5])

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.gcf().subplots_adjust(bottom=0.26)
plt.show()
fig.savefig('Feature importances of determining Genre.png',dpi=220);


# In[49]:


import pandas as pd
data3 = pd.read_csv('direct_influence_no_filter.csv')
data3
data4= pd.read_csv('data_by_artist.csv')
data3
data5=pd.merge(data3,data4,left_on='influencer_id',right_on='artist_id')
data6=data5
data5
data5['tempo_new']=0
data5['loudness_new']=0
data5['duration_ms_new']=0

data5['tempo_new']=(data5['tempo']-data5['tempo'].min())/(data5['tempo'].max()-data5['tempo'].min())
data5['loudness_new']=(data5['loudness']-data5['loudness'].min())/(data5['loudness'].max()-data5['loudness'].min())
data5['duration_ms_new']=(data5['duration_ms']-data5['duration_ms'].min())/(data5['duration_ms'].max()-data5['duration_ms'].min())

data5= data5.drop(['mode','influencer_id','influencer_name','artist_name',"artist_id","tempo",'loudness','duration_ms','popularity','count','key'], 1)
data5


# In[50]:


data5['count(*)'].describe()


# In[53]:


'''
Random Forest--predicttion of comment_num,like_num,repost_num
'''
X = data5.drop(['count(*)'],axis=1)
bins = [0,3 , 5, 15,615]
y = pd.cut(data5['count(*)'], bins,labels=[1,2,3,4],right=True)

#import packages needed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = RandomForestClassifier(n_estimators=500, random_state=10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))
importances = classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    
# Plot the feature importances of the forest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection

get_ipython().run_line_magic('matplotlib', 'inline')

fig= plt.figure()
ax=plt.axes()

map_vir = cm.get_cmap(name='YlGn')
n=5
norm = plt.Normalize(importances[indices][:n].min(), importances[indices][:n].max())
norm_y = norm(importances[indices][:n])
color = map_vir(norm_y)

plt.title("Most Contagious Features")

plt.bar(range(5), importances[indices][:5],
       color=color, yerr=std[indices][:5], align="center")
plt.xticks(range(5), ['Instrumentalness','Energy','Loudness','Liveness','Danceability'])

plt.xlim([-1, 5])

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.gcf().subplots_adjust(bottom=0.26)
plt.show()
fig.savefig('contagious.png',dpi=220);


# In[52]:


pd.DataFrame(data5.corr('spearman'))


# In[ ]:




