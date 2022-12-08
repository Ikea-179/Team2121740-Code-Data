#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv('artist_increase_genre_year.csv')
data


# In[2]:


data['genre'].value_counts()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
rng = np.random.RandomState(0)

x=np.array(data['active_start'])
y=np.array(data['count(*)'])
c=np.array(data['genre'])
plt.scatter(x, y)


# In[4]:


x


# In[5]:


df=data
df.describe()
genres = np.sort(data['genre'].dropna().unique())
fig, axes = plt.subplots(1, 20, figsize=(60, 40), sharey=True)

for ax, genre in zip(axes, genres):
    data6= data[data['genre']==genre]
    x=np.array(data6['active_start'])
    y=np.array(data6['count(*)'])
    ax.scatter(x, y)
    ax.set(title=genre)
   


# In[6]:


df=data
df.describe()
genres = np.sort(data['genre'].dropna().unique())
fig, axes = plt.subplots(1, 1, figsize=(15, 10), sharey=True)
#plt.style.use('seaborn')

for genre in genres:
    data6= data[data['genre']==genre]
    data6=data6.sort_values(by=['active_start'])
    x=np.array(data6['active_start'])
    y=np.array(data6['count(*)'])
    plt.plot(x, y, label=genre)

axes.legend()
fig.savefig('increase per year(with).png',dpi=200);


# In[7]:


df=data
df.describe()
data7=data[data['genre']!='Pop/Rock']
genres = np.sort(data7['genre'].dropna().unique())
fig, axes = plt.subplots(1, 1, figsize=(15, 10), sharey=True)
plt.style.use('seaborn')

for genre in genres:
    data6= data[data['genre']==genre]
    data6=data6.sort_values(by=['active_start'])
    x=np.array(data6['active_start'])
    y=np.array(data6['count(*)'])
    plt.plot(x, y, label=genre)
axes.legend()
fig.savefig('increase per year(without).png',dpi=220);


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(60, 40))
ax=plt.axes()

data.groupby('genre')['active_start'].plot(kind='kde', legend=True, figsize=(10, 5))


# In[9]:


pop=data.loc[data['genre']=='Pop/Rock']
pop=pop.sort_values(by=['active_start'])

classical=data.loc[data['genre']=='Classical']
classical=classical.sort_values(by=['active_start'])

rb=data.loc[data['genre']=='R&B;']
rb=rb.sort_values(by=['active_start'])


# In[25]:


x1=np.array(pop['active_start'])
y1=np.array(pop['count(*)'])

x2=np.array(classical['active_start'])
y2=np.array(classical['count(*)'])

x3=np.array(rb['active_start'])
y3=np.array(rb['count(*)'])

fig, ax = plt.subplots()

plt.style.use('seaborn-white')

# Using set_dashes() to modify dashing of an existing line
line1, = ax.plot(x1, y1, label='The Beatles',c='royalblue')

# Using plot(..., dashes=...) to set the dashing when creating a line
line2, = ax.plot(x2, y2,  label='Bob Dylan',c= 'c')

# Using plot(..., dashes=...) to set the dashing when creating a line
line3, = ax.plot(x3, y3, label='Nirvana',c='tab:green')

# Using plot(..., dashes=...) to set the dashing when creating a line
line4, = ax.plot(x3, y3, label='Neil Young',c='y')

line5, = ax.plot(x3, y3, label='David Bowie',c='blueviolet')



ax.legend()
plt.show()
fig.savefig('颜色2.png',dpi=220);


# In[20]:


rb


# In[12]:


x1=np.array(pop['active_start'])
y1=np.array(pop['count(*)'])

x2=np.array(classical['active_start'])
y2=np.array(classical['count(*)'])

x3=np.array(rb['active_start'])
y3=np.array(rb['count(*)'])

fig, ax = plt.subplots()

# Using set_dashes() to modify dashing of an existing line
line1, = ax.plot(x1, y1, label='Pop/Rock',c='forestgreen')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

# Using plot(..., dashes=...) to set the dashing when creating a line
line2, = ax.plot(x2, y2, dashes=[6, 2], label='Classical',c='gold')

# Using plot(..., dashes=...) to set the dashing when creating a line
line3, = ax.plot(x3, y3, label='R&B',c='steelblue')
line3.set_dashes([1,1,1,1,1,1]) 

ax.legend()
fig.savefig('increase per year.png',dpi=220);
plt.show()


# In[ ]:




