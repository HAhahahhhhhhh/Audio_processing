from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import butter, lfilter

# Function to apply a high-pass filter
def highpass_filter(data, cutoff=300, fs=22050, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)

# Load the example clip
y, sr = librosa.load('examples/audio/Kevin_MacLeod_-_Vibe_Ace.mp3', offset=40, duration=10)

# Normalize the audio signal
y = y / np.max(np.abs(y))

# Apply high-pass filter
y_filtered = highpass_filter(y, cutoff=300, fs=sr)

# Compute the short-time Fourier transform of the filtered signal
D = librosa.stft(y_filtered)

# Decompose D into harmonic and percussive components
D_harmonic, D_percussive = librosa.decompose.hpss(D)

# Let's compute separations for a few different margins and compare the results
margins = [1, 2, 4, 8, 16]
D_harmonic_list = []
D_percussive_list = []

for margin in margins:
    D_harmonic_temp, D_percussive_temp = librosa.decompose.hpss(D, margin=margin)
    D_harmonic_list.append(D_harmonic_temp)
    D_percussive_list.append(D_percussive_temp)

# Visualization
rp = np.max(np.abs(D))

plt.figure(figsize=(12, 10))

plt.subplot(len(margins) + 1, 2, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=rp), y_axis='log')
plt.colorbar()
plt.title('Full Spectrogram')

for i, margin in enumerate(margins):
    plt.subplot(len(margins) + 1, 2, (i + 2) * 2 - 1)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_harmonic_list[i]), ref=rp), y_axis='log')
    plt.title(f'Harmonic (margin={margin})')
    plt.yticks([])

    plt.subplot(len(margins) + 1, 2, (i + 2) * 2)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_percussive_list[i]), ref=rp), y_axis='log')
    plt.title(f'Percussive (margin={margin})')
    plt.yticks([])

plt.tight_layout()
plt.show()
