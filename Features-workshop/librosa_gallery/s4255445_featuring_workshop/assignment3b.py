import numpy as np
import librosa
import librosa.display
from scipy.signal import butter, lfilter
import soundfile as sf
import matplotlib.pyplot as plt

# High-pass filter to remove low frequencies
def highpass_filter(data, cutoff=200, fs=22050, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)

# Load the audio file
y, sr = librosa.load('examples/audio/track04_sour_le_vent.mp3', duration=120)

# Preprocess audio with high-pass filter
y_filtered = highpass_filter(y, cutoff=200, fs=sr)

# Compute the STFT for the filtered audio
S_full, phase = librosa.magphase(librosa.stft(y_filtered))

# Apply non-local filtering with an adaptive frame width for vocal clarity
frame_width = int(librosa.time_to_frames(2, sr=sr))
S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=frame_width)
S_filter = np.minimum(S_full, S_filter)

# Apply enhanced soft-masking with a higher margin for vocals
margin_i, margin_v = 2, 15  # Increase margin_v for clearer separation of vocals
power = 2
mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)
mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)

# Separate foreground (vocals) and background (instrumental)
S_foreground = mask_v * S_full
S_background = mask_i * S_full

# Convert the foreground (vocals) spectrogram back to time-domain signal
y_vocals = librosa.istft(S_foreground * phase)

# Save the separated audio to files
sf.write('track04_byAdapt.wav', y_vocals, sr)

# Optional: Save the instrumental track as well
y_instrumental = librosa.istft(S_background * phase)
sf.write('track04_instrumental_only.wav', y_instrumental, sr)

# Plotting spectrogram for vocals only (optional visualization step)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(S_foreground, ref=np.max), y_axis='log', x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of Separated Vocals')
plt.tight_layout()
plt.show()
