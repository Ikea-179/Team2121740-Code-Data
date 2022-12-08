#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#load the raw data
#data = pd.read_csv('data_by_year.csv')
data = pd.read_csv('num_influencer_distribution.csv')


# In[36]:


data


# In[38]:


x=np.array(data['num_influencers'])
y=np.array(data['Unnamed: 2'])


fig, ax = plt.subplots()
plt.style.use('seaborn')
# Using set_dashes() to modify dashing of an existing line
line1, = ax.plot(x, y,c='black')
#line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

plt.xlabel('Number of Influencers')
plt.ylabel('Counts')
fig.savefig('num_influencers.png',dpi=200);
plt.show()


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#load the raw data
#data = pd.read_csv('data_by_year.csv')
data = pd.read_csv('music.csv')


# In[6]:


data


# In[10]:



data['tempo_new']=0
data['loudness_new']=0
data['duration_ms_new']=0

data['tempo_new']=(data['tempo']-data['tempo'].min())/(data['tempo'].max()-data['tempo'].min())
data['loudness_new']=(data['loudness']-data['loudness'].min())/(data['loudness'].max()-data['loudness'].min())
data['duration_ms_new']=(data['duration_ms']-data['duration_ms'].min())/(data['duration_ms'].max()-data['duration_ms'].min())

data1= data.drop(['Unnamed: 0','artists_id',"tempo",'loudness','duration_ms','popularity','release_date'], 1)
data1['artist_names'].count()


# In[20]:


data2=data1[data1['artist_names']=='The Beatles']
data2


# In[22]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#load the raw data
#data = pd.read_csv('data_by_year.csv')
df= pd.read_csv('pop人换爹历史.csv')


# In[23]:


d1960=df[df['follower_active_start']==1960.0]
d1960


# In[24]:


d1970=df[df['follower_active_start']==1970.0]
d1970


# In[26]:


d1980=df[df['follower_active_start']==1980.0]
d1980


# In[28]:


d1990=df[df['follower_active_start']==1990.0]
d1990


# In[29]:


d2000=df[df['follower_active_start']==2000.0]
d2000


# In[30]:


d2010=df[df['follower_active_start']==2010.0]
d2010


# In[ ]:





# In[ ]:




