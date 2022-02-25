import IPython
import numpy as np
import wave
import sklearn

mix_1_wave = wave.open('./music/mixedX.wav','r')
signal_1_raw = mix_1_wave.readframes(-1)
signal_1 = np.frombuffer(signal_1_raw,dtype=np.int16)

mix_2_wave = wave.open('./music/mixedY.wav','r')
signal_raw_2 = mix_2_wave.readframes(-1)
signal_2 = np.frombuffer(signal_raw_2, np.int16)

X= list(zip(signal_1, signal_2))
from sklearn.decomposition import FastICA

ica = FastICA(n_components=2)
ica_result = ica.fit_transform(X)

result_signal_1 = ica_result[:,0]
result_signal_2 = ica_result[:,1]

from scipy.io import wavfile
result_signal_1_int = np.int16(result_signal_1*32767*100)
result_signal_2_int = np.int16(result_signal_2*32767*100)

wavfile.write("./music/result_signal_1.wav", fs, result_signal_1_int)
wavfile.write("./music/result_signal_2.wav", fs, result_signal_2_int)

IPython.display.Audio("./music/result_signal_1.wav")
IPython.display.Audio("./music/result_signal_2.wav")





