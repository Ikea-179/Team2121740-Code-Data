#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#load the raw data
#data = pd.read_csv('data_by_year.csv')
data = pd.read_csv('full_music_data.csv')
data


# In[2]:


data['tempo_new']=0
data['loudness_new']=0
data['duration_ms_new']=0

data['tempo_new']=(data['tempo']-data['tempo'].min())/(data['tempo'].max()-data['tempo'].min())
data['loudness_new']=(data['loudness']-data['loudness'].min())/(data['loudness'].max()-data['loudness'].min())
data['duration_ms_new']=(data['duration_ms']-data['duration_ms'].min())/(data['duration_ms'].max()-data['duration_ms'].min())

data1= data.drop(['artist_names',"tempo",'loudness','popularity','release_date','explicit'], 1)
data1


# In[44]:


data2=data1[data1['artists_id']=='[754032]']
data3=data2[(data2['year']>=1960) & (data2['year']<1970)]
data3


# In[45]:


data4=data3.describe()


# In[46]:


data3['mode'].sum()


# In[48]:


data3['duration_ms'].describe()


# In[ ]:





# In[67]:


data5=data4.drop(['mode','year','key','duration_ms','duration_ms_new'],1)
data5=data5.loc['mean'].tolist()
data5


# In[68]:


data6=data1[data1['artists_id']=='[66915]']
data7=data6[(data6['year']>=1960) & (data6['year']<1970)]
data8=data7.drop(['mode','year','key','duration_ms','duration_ms_new'],1)
data8=data8.describe()
data9=data8.loc['mean']
data9


# In[95]:


data8


# In[66]:


data61=data1[data1['artists_id']=='[894465]']
data71=data61[(data61['year']>=1960) & (data61['year']<1970)]
data81=data71.drop(['mode','year','key','duration_ms','duration_ms_new'],1)
data81=data81.describe()
data91=data81.loc['mean'].tolist()
data91


# In[96]:


data62=data1[data1['artists_id']=='[754032]']
data72=data62[(data62['year']>=1970) & (data62['year']<1980)]
data82=data72.drop(['mode','year','key','duration_ms','duration_ms_new'],1)
data82=data82.describe()
data92=data82.loc['mean'].tolist()
data92


# In[97]:


data63=data1[data1['artists_id']=='[894465]']
data73=data63[(data63['year']>=1970) & (data63['year']<1980)]
data83=data73.drop(['mode','year','key','duration_ms','duration_ms_new'],1)
data83=data83.describe()
data93=data83.loc['mean'].tolist()
data93


# In[98]:


data64=data1[data1['artists_id']=='[418740]']
data74=data64[(data64['year']>=1970) & (data64['year']<1980)]
data84=data74.drop(['mode','year','key','duration_ms','duration_ms_new'],1)
data84=data84.describe()
data94=data84.loc['mean'].tolist()
data94


# In[126]:


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def example_data():
    # The following data is from the Denver Aerosol Sources and Health study.
    # See doi:10.1016/j.atmosenv.2008.12.017
    #
    # The data are pollution source profile estimates for five modeled
    # pollution sources (e.g., cars, wood-burning, etc) that emit 7-9 chemical
    # species. The radar charts are experimented with here to see if we can
    # nicely visualize how the modeled source profiles change across four
    # scenarios:
    #  1) No gas-phase species present, just seven particulate counts on
    #     Sulfate
    #     Nitrate
    #     Elemental Carbon (EC)
    #     Organic Carbon fraction 1 (OC)
    #     Organic Carbon fraction 2 (OC2)
    #     Organic Carbon fraction 3 (OC3)
    #     Pyrolized Organic Carbon (OP)
    #  2)Inclusion of gas-phase specie carbon monoxide (CO)
    #  3)Inclusion of gas-phase specie ozone (O3).
    #  4)Inclusion of both gas-phase species is present...
    data = [
        ['danceability','energy','valence','acousticness','instrumentalness','liveness','speechiness','tempo','loudness'],
        ('1960s', [
           data5,
           data9,
           data91,
        [0,0,0,0,0,0,0,0,0]]),
        ('1970s', [
          data92,
            [0,0,0,0,0,0,0,0,0],
           data93,
           data94]),


    ]
    return data


if __name__ == '__main__':
    N = 9
    theta = radar_factory(N, frame='polygon')

    data = example_data()
    spoke_labels = data.pop(0)

    fig, axs = plt.subplots(figsize=(9, 9), nrows=1, ncols=2,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.4, hspace=0.20, top=1.4, bottom=0.05)
    plt.style.use('seaborn')

    colors = ['royalblue', 'c', 'tab:green','y']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.05)
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    labels = ('Factor 1', 'Factor 2', 'Factor 3')

    fig.text(0.5, 0.965, '',
             horizontalalignment='center', color='black', weight='bold',
             size='large')

    plt.show()
    fig.savefig('leida.png',dpi=150);


# In[33]:


data65=data1[data1['artists_id']=='[754032]']
data75=data65[(data65['year']>=1990) & (data65['year']<2000)]
data85=data75.drop(['mode','year','key','duration_ms','duration_ms_new'],1)
data85=data85.describe()
data95=data85.loc['mean'].tolist()
data95


# In[34]:


data66=data1[data1['artists_id']=='[379125]']
data76=data66[(data66['year']>=1990) & (data66['year']<2000)]
data86=data76.drop(['mode','year','key','duration_ms','duration_ms_new'],1)
data86=data86.describe()
data96=data86.loc['mean'].tolist()
data96


# In[35]:


data67=data1[(data1['artists_id']=='[379125, 130932]')| (data1['artists_id']=='[379125]')|(data1['artists_id']=='[38490, 379125]') |(data1['artists_id']=='[66915, 834466, 612716, 379125, 187478, 209142]')]
data77=data67[(data67['year']>=1990) & (data67['year']<2000)]
data87=data77.drop(['mode','year','key','duration_ms','duration_ms_new'],1)
data87=data87.describe()
data97=data87.loc['mean'].tolist()
data97


# In[42]:


data651=data1[data1['artists_id']=='[754032]']
data751=data651[(data651['year']>=2000) & (data651['year']<2010)]
data851=data751.drop(['mode','year','key','duration_ms','duration_ms_new'],1)
data851=data851.describe()
data951=data851.loc['mean'].tolist()
data951


# In[43]:


data652=data1[data1['artists_id']=='[66915]']
data752=data652[(data652['year']>=2000) & (data652['year']<2010)]
data852=data752.drop(['mode','year','key','duration_ms','duration_ms_new'],1)
data852=data852.describe()
data952=data852.loc['mean'].tolist()
data952


# In[44]:


data653=data1[data1['artists_id']=='[531986]']
data753=data653[(data653['year']>=2000) & (data653['year']<2010)]
data853=data753.drop(['mode','year','key','duration_ms','duration_ms_new'],1)
data853=data853.describe()
data953=data853.loc['mean'].tolist()
data953


# In[ ]:





# In[50]:


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def example_data():
    # The following data is from the Denver Aerosol Sources and Health study.
    # See doi:10.1016/j.atmosenv.2008.12.017
    #
    # The data are pollution source profile estimates for five modeled
    # pollution sources (e.g., cars, wood-burning, etc) that emit 7-9 chemical
    # species. The radar charts are experimented with here to see if we can
    # nicely visualize how the modeled source profiles change across four
    # scenarios:
    #  1) No gas-phase species present, just seven particulate counts on
    #     Sulfate
    #     Nitrate
    #     Elemental Carbon (EC)
    #     Organic Carbon fraction 1 (OC)
    #     Organic Carbon fraction 2 (OC2)
    #     Organic Carbon fraction 3 (OC3)
    #     Pyrolized Organic Carbon (OP)
    #  2)Inclusion of gas-phase specie carbon monoxide (CO)
    #  3)Inclusion of gas-phase specie ozone (O3).
    #  4)Inclusion of both gas-phase species is present...
    data = [
        ['danceability','energy','valence','acousticness','instrumentalness','liveness','speechiness','tempo','loudness'],
        ('1990s', [
           data97,
           data95,
           data96,
            [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0]]),
        ('2000s', [
            [0,0,0,0,0,0,0,0,0],
          data951,
            [0,0,0,0,0,0,0,0,0],
           data952,
           data953]),


    ]
    return data


if __name__ == '__main__':
    N = 9
    theta = radar_factory(N, frame='polygon')

    data = example_data()
    spoke_labels = data.pop(0)

    fig, axs = plt.subplots(figsize=(9, 9), nrows=1, ncols=2,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.4, hspace=0.20, top=1.4, bottom=0.05)
    plt.style.use('seaborn')

    colors = ['tab:green','royalblue', 'y','c','blueviolet']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.05)
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    labels = ('Factor 1', 'Factor 2', 'Factor 3')

    fig.text(0.5, 0.965, '',
             horizontalalignment='center', color='black', weight='bold',
             size='large')

    plt.show()
    fig.savefig('leida2.png',dpi=150);


# In[32]:





# In[ ]:




