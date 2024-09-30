import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.image import resize
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical
import seaborn as sns
# Load the model and classes list
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("C:\\Users\\Admin\\OneDrive\\Desktop\\Music Genre Classification System\\Trained_model.h5")  # Update with your model path
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    return model, classes

# Load and preprocess audio data
def load_and_preprocess_file(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
                
    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    # Iterate over each chunk
    for i in range(num_chunks):
        # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
                    
        # Extract the chunk of audio
        chunk = audio_data[start:end]
                    
        # Compute the Mel-spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                    
        # Resize the Mel-spectrogram to match model input size
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)

# Model prediction function
def model_prediction(X_test, model, classes):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    
    # Get the most frequent predicted category
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    
    return classes[max_elements[0]]

# Function to plot waveform
def plot_waveform(y, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    st.pyplot(plt)

# Function to plot spectrogram
def plot_spectrogram(y, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (dB)')
    st.pyplot(plt)

# Streamlit app interface
st.title("Music Genre Classification App")

# File uploader for audio files
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

# Load the model and class labels
model, classes = load_model()

# Process the uploaded audio file
if uploaded_file is not None:
    if st.button('Show'):
        # Save uploaded file temporarily
        with open("temp_audio_file.mp3", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_path = "temp_audio_file.mp3"
        
        # Load and display the audio
        y, sr = librosa.load(file_path, sr=44100)
        st.audio(uploaded_file, format='audio/mp3')
        
        # Plot waveform
        st.subheader("Waveform")
        plot_waveform(y, sr)
        
        # Plot spectrogram
        st.subheader("Spectrogram")
        plot_spectrogram(y, sr)
        
        # Extract features for classification (split into chunks)
        st.subheader("Classifying the Genre...")
        X_test = load_and_preprocess_file(file_path)

        # Show the shape of extracted features for debugging
        st.write("Processed features shape:", X_test.shape)

        # Make predictions
        predicted_genre = model_prediction(X_test, model, classes)

        # Display the predicted genre
        st.subheader(f"Predicted Genre: {predicted_genre}")
