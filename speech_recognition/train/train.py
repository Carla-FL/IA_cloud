import librosa
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Function to extract MFCC features from an audio file
def extract_features(audio_path, max_pad_len=40):
    # Load the audio file
    audio, sample_rate = librosa.load(audio_path, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    
    # Check if the number of time steps is less than the max_pad_len
    time_steps = mfccs.shape[1]
    if time_steps < max_pad_len:
        # If fewer time steps, pad the MFCC matrix
        pad_width = max_pad_len - time_steps
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        # If more time steps, truncate to max_pad_len
        mfccs = mfccs[:, :max_pad_len]
    
    return mfccs.T  # Transpose to shape (time_steps, features)

# Example audio files (replace with your audio files)
audio_files = ['audio1.wav', 'audio2.wav']

# Extract features from all audio files
X = np.array([extract_features(file) for file in audio_files])
y = np.array([0, 1])  # Labels for audio files


'''model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(40, 20)))  # 40 time steps, 20 features
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()'''

# Define the model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(100, 20)),  # (timesteps, features)
    LSTM(128),
    Dense(64, activation="relu"),
    #Dense(10, activation="softmax")  # Assume 10 possible words to recognize
    Dense(1, activation='sigmoid')  # Binary classification
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()


model.fit(X, y, epochs=1000, batch_size=16)

# Save the entire model
model.save("lstm_model.h5")
print("Model saved successfully!")