#!/usr/bin/env python
# coding: utf-8

# #                             Voice Separation

# In[55]:


import os
print (os.getcwd() )
os.chdir("E:\Cocktail-Party-Problem-master")


# In[56]:


get_ipython().system('pip install pydub')
from pydub import AudioSegment
import IPython
import numpy as np
import wave

mix_1_wave = wave.open('./sounds/mixedX.wav','r')


# In[57]:


mix_1_wave.getparams()


# In[58]:


294720/48000


# Let's extract the frames of the wave file, which will be a part of the dataset we'll run ICA against:

# In[59]:


# Extract Raw Audio from Wav File
signal_1_raw = mix_1_wave.readframes(-1)


# In[60]:


#signal_1_raw


# In[61]:


signal_1 = np.frombuffer(signal_1_raw,dtype=np.int16)
signal_1
signal_1.size


# signal_1 is now a list of ints representing the sound contained in the first file.

# In[62]:


'length: ', len(signal_1) , 'first 100 elements: ',signal_1[:200]


# In[63]:


import matplotlib.pyplot as plt

fs = mix_1_wave.getframerate()
timing = np.linspace(0, len(signal_1)/fs, num=len(signal_1))


plt.figure(figsize=(12,2))
plt.title('Recording 1')
plt.plot(timing,signal_1, c="#3ABFE7")
plt.ylim(-35000, 35000)
plt.show()


# You can hear this recoding below

# In the same way, we can now load the other two wave files, and hear then...

# In[64]:



mix_2_wave = wave.open('./sounds/mixedY.wav','r')

#Extract Raw Audio from Wav File
signal_raw_2 = mix_2_wave.readframes(-1)


# In[65]:


signal_2 = np.frombuffer(signal_raw_2, np.int16)


# In[66]:


plt.figure(figsize=(12,2))
plt.title('Recording 2')
plt.plot(timing,signal_2, c="#3ABFE7")
plt.ylim(-35000, 35000)
plt.show()


# # Listening Audio

# In[67]:


IPython.display.Audio("./sounds/mixedX.wav")


# In[68]:


IPython.display.Audio("./sounds/mixedY.wav")


# ### So you can clearly hear that the recording seems to have multiple different recordings combined

# In[69]:


X= list(zip(signal_1, signal_2))
#X2= (zip(signal_1, signal_2))


# # Applying FastICA

# In[70]:


import sklearn
from sklearn.decomposition import FastICA

# Initializing FastICA with n_components=3
ica = FastICA(n_components=2)


# In[71]:



# Running the FastICA algorithm using fit_transform on dataset X
ica_result = ica.fit_transform(X)


# In[72]:


ica_result.shape


# Let's split into separate signals and look at them

# In[73]:


result_signal_1 = ica_result[:,0]
result_signal_2 = ica_result[:,1]


# # Results
# 
# Let's plot to see how the wave forms look

# In[74]:


# Independent Component #1
plt.figure(figsize=(12,2))
plt.title('Independent Component #1')
plt.plot(result_signal_1, c="#df8efd")
plt.ylim(-0.010, 0.010)
plt.show()

# Independent Component #2
plt.figure(figsize=(12,2))
plt.title('Independent Component #2')
plt.plot(result_signal_2, c="#87de72")
plt.ylim(-0.010, 0.010)
plt.show()




# In[75]:


from scipy.io import wavfile

# Converting to int, mapping the appropriate range, and increasing the volume a little bit
result_signal_1_int = np.int16(result_signal_1*32767*100)
result_signal_2_int = np.int16(result_signal_2*32767*100)



# Writing wave files
wavfile.write("./sounds/result_signal_1.wav", fs, result_signal_1_int)
wavfile.write("./sounds/result_signal_2.wav", fs, result_signal_2_int)


# In[76]:


IPython.display.Audio("./sounds/result_signal_1.wav")


# In[77]:


IPython.display.Audio("./sounds/result_signal_2.wav")


# In[ ]:




