import librosa

# Load the audio for 'Vibe_Ace.mp3'
vibeace_path = 'examples/audio/Kevin_MacLeod_-_Vibe_Ace.mp3'
y_vibeace, sr_vibeace = librosa.load(vibeace_path)
tempo_vibeace, _ = librosa.beat.beat_track(y=y_vibeace, sr=sr_vibeace)

# Load the audio for one other track
track_path01 = 'examples/audio/track01_ErosRamazotti.mp3'
y_track, sr_track = librosa.load(track_path01)
tempo_track01, _ = librosa.beat.beat_track(y=y_track, sr=sr_track)

track_path02 = 'examples/audio/track02_cormac_begley_start.mp3'
y_track, sr_track = librosa.load(track_path02)
tempo_track02, _ = librosa.beat.beat_track(y=y_track, sr=sr_track)

track_path03 = 'examples/audio/track03_rolling_stone_blues_end.mp3'
y_track, sr_track = librosa.load(track_path03)
tempo_track03, _ = librosa.beat.beat_track(y=y_track, sr=sr_track)

track_path04 = 'examples/audio/track04_sour_le_vent.mp3'
y_track, sr_track = librosa.load(track_path04)
tempo_track04, _ = librosa.beat.beat_track(y=y_track, sr=sr_track)

# Convert tempos to float
tempo_vibeace = float(tempo_vibeace)
tempo_track01 = float(tempo_track01)
tempo_track02 = float(tempo_track02)
tempo_track03 = float(tempo_track03)
tempo_track04 = float(tempo_track04)

print('Estimated tempo for Vibe Ace: {:.2f} BPM'.format(tempo_vibeace))
print('Estimated tempo for track01: {:.2f} BPM'.format(tempo_track01))
print('Estimated tempo for track02: {:.2f} BPM'.format(tempo_track02))
print('Estimated tempo for track03: {:.2f} BPM'.format(tempo_track03))
print('Estimated tempo for track04: {:.2f} BPM'.format(tempo_track04))
