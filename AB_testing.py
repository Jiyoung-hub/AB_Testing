#!/usr/bin/env python
# coding: utf-8

# # read data

# In[1]:


import pandas as pd


# In[2]:


from scipy.stats import norm


# In[3]:


df = pd.read_csv("AB_test_data.csv")


# In[4]:


df.head()


# # alpha, power  

# In[5]:


alpha = 0.05
power = 0.8
norm.ppf(1-alpha)


# In[6]:


norm.ppf(1-alpha/2)


# In[7]:


norm.ppf(1-power)


# In[8]:


df.purchase_TF.value_counts()


# # Overview data

# In[9]:


df[df.Variant == 'A'].purchase_TF.count()


# In[10]:


df[df.Variant == 'B'].purchase_TF.count()


# In[11]:


df[df.Variant == 'A'].purchase_TF.value_counts()


# In[12]:


df[df.Variant == 'B'].purchase_TF.value_counts()


# # Calculating the optimal sample size

# In[13]:


from numpy import sqrt


# In[14]:


alpha = 0.05
power = 0.8
t_alpha_d2 = norm.ppf(1-alpha/2)
t_alpha_d2
t_beta = .84162
p0 = df[df.Variant == 'A'].purchase_TF.sum() / 50000
p1 = df[df.Variant == 'B'].purchase_TF.sum() / 5000
p_bar = (p0  + p1)/2
delta = p1  - p0


# In[15]:


t_alpha_d2


# In[16]:


((t_alpha_d2*sqrt(2*p_bar*(1-p_bar)))+t_beta*sqrt((p0*(1-p0))+(p1*(1-p1))))**2/delta/delta


# # z_score 

# In[17]:


df_A= df[df.Variant=='A']


# In[18]:


df_B = df[df.Variant=='B']


# In[19]:


z = (0.1962-0.15206) / sqrt(0.15206*(1-0.15206)/5000)
z


# # Calculating z_score for 10 trials

# In[20]:


def z_score(data,number,p=0.15206):
    
    variant_B = data
    
    variant_B_sampled = variant_B.sample(n = number)
    
    p_sample = variant_B_sampled.purchase_TF.sum() / number

    z_score = (p_sample-p) / sqrt(p*(1-p)/number)

    return z_score


# In[21]:


z_score(df_B,1157,p=0.15206)


# In[22]:


z_score(df_B,1157,p=0.15206)


# In[23]:


z_score(df_B,1157,p=0.15206)


# In[24]:


z_score(df_B,1157,p=0.15206)


# In[25]:


z_score(df_B,1157,p=0.15206)


# In[26]:


z_score(df_B,1157,p=0.15206)


# In[27]:


z_score(df_B,1157,p=0.15206)


# In[28]:


z_score(df_B,1157,p=0.15206)


# In[29]:


z_score(df_B,1157,p=0.15206)


# In[30]:


z_score(df_B,1157,p=0.15206)


# # Sequential probability ratio test (SPRT)

# In[31]:


"""
p(xi=1)= 0.15206 under H0
p(xi=1)= 0.1962 under H1
a=.05
b=.2
"""


# In[32]:


import numpy as np
np.log(.2)


# In[33]:


def test(data,number,p_ho,p_h1):
    
    variant_B = data
    
    variant_B_sampled = variant_B.sample(n = number)

    total_sum=0
    i=0
    while -1.6 < total_sum <  2.99:     
        if variant_B_sampled.purchase_TF.iloc[i] == True:
            log = np.log(p_h1/p_ho)
        elif variant_B_sampled.purchase_TF.iloc[i] == False:
            log = np.log((1-p_h1)/(1-p_ho))
        total_sum = total_sum + log
        i = i+1
    return i, total_sum


# In[34]:


test(df_B,1158,p_ho=0.15206,p_h1=0.1962)


# In[35]:


test(df_B,1158,p_ho=0.15206,p_h1=0.1962)


# In[36]:


test(df_B,1158,p_ho=0.15206,p_h1=0.1962)


# In[37]:


test(df_B,1158,p_ho=0.15206,p_h1=0.1962)


# In[38]:


test(df_B,1158,p_ho=0.15206,p_h1=0.1962)


# In[39]:


test(df_B,1158,p_ho=0.15206,p_h1=0.1962)


# In[40]:


test(df_B,1158,p_ho=0.15206,p_h1=0.1962)


# In[41]:


test(df_B,1158,p_ho=0.15206,p_h1=0.1962)


# In[42]:


test(df_B,1158,p_ho=0.15206,p_h1=0.1962)


# In[43]:


test(df_B,1158,p_ho=0.15206,p_h1=0.1962)


# In[45]:


import numpy as np
p_ho=0.15206
p_h1=0.1962
i=0
total_sum = 0
while -1.6 < total_sum < 2.99:
    if df_B.purchase_TF.iloc[i] == True:
        log = np.log(p_h1/p_ho)
    elif df_B.purchase_TF.iloc[i] == False:
        log = np.log((1-p_h1)/(1-p_ho))
    total_sum = total_sum + log
    print(i, total_sum)
    i = i+1

