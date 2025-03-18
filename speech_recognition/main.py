import librosa
import numpy as np
import os
from tensorflow.keras.models import  load_model

import sounddevice as sd
import numpy as np
import wave

 

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


# Load the model
model = load_model("./train/lstm_model.h5")
print("Model loaded successfully!")

while True:

    input("Commencer l'enregistrement...")

    # Settings
    samplerate = 44100  # Sample rate in Hz
    duration = 3  # Duration in seconds
    filename = "output.wav"

    print("Recording...")

    # Record audio
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=2, dtype=np.int16)
    sd.wait()  # Wait for recording to finish

    print("Recording complete. Saving file...")

    # Save as WAV file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(2)  # Stereo
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    print(f"File saved as {filename}")



    # Load and extract features from a new audio file
    new_audio_file = 'output.wav'
    new_features = extract_features(new_audio_file)

    # Reshape for prediction (add batch dimension)
    new_features = np.expand_dims(new_features, axis=0) # Add batch dimension

    # Predict
    '''prediction = model.predict(new_features)
    predicted_class = 1 if prediction > 0.5 else 0

    print("Prediction Probability:", prediction)
    print("Predicted Class:", predicted_class)'''
   
    prediction = model.predict(new_features)
    predicted_word = np.argmax(prediction)

    print(f"Predicted Word Index: {predicted_word}")