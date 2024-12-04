# -*- coding: utf-8 -*-
"""
================
Vocal separation
================

This script demonstrates separating vocals from accompanying instrumentation
using the "REPET-SIM" method with some additional adjustments for better separation.
"""

# Imports
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf  # Import soundfile for saving audio

#############################################
# Load an example audio file with vocals.
y, sr = librosa.load('examples/audio/track04_sour_le_vent.mp3', duration=120)

# Compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))

# Use a non-local filter to reduce sparse, non-repetitive deviations (e.g., vocals)
S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
S_filter = np.minimum(S_full, S_filter)

# Apply soft masking with different margins to create foreground and background masks
margin_i, margin_v = 2, 10
power = 2
mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)
mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)

# Separate the spectrograms for vocals (foreground) and background
S_foreground = mask_v * S_full
S_background = mask_i * S_full

# Reconstruct the waveform of the vocal part only
y_vocals = librosa.istft(S_foreground * phase)

# Save the vocals as a .wav file
sf.write('track04_vocals_only.wav', y_vocals, sr)

##########################################
# Plot and visualize results for verification
idx = slice(*librosa.time_to_frames([30, 35], sr=sr))
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max), y_axis='log', sr=sr)
plt.title('Full spectrum')
plt.colorbar()

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max), y_axis='log', sr=sr)
plt.title('Background')
plt.colorbar()

plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max), y_axis='log', sr=sr)
plt.title('Foreground (Vocals)')
plt.colorbar()
plt.tight_layout()
plt.show()
