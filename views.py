from pyexpat.errors import messages
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.http import JsonResponse
import joblib
import pandas as pd
from django.contrib import messages
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D, LSTM, Concatenate, Dense, Input
from tensorflow.keras.models import Model


def home(request):
    return render(request, "home.html")

def register(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Registration successful. You can now log in.")
            return redirect("login")  # Redirect to the login page after successful registration
        else:
            # Display form errors explicitly
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field.capitalize()}: {error}")
    else:
        form = UserCreationForm()
    
    return render(request, "register.html", {"form": form})


def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect("upload_sentiment_data")  
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field.capitalize()}: {error}")
    else:
        form = AuthenticationForm()

    return render(request, "login.html", {"form": form})

def logout_view(request):
    logout(request)
    return redirect("home")

def dashboard(request):
    
    return render(request,'upload_form.html')

from .forms import SentimentDataForm
from .models import SentimentData

# Parameters
image_size = (224, 224)
max_caption_length = 100

tokenizer_path = r"F:\Multimodal\tokenizer1.pkl"  # Ensure this file exists
if os.path.exists(tokenizer_path):
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)
    vocab_size = len(tokenizer.word_index) + 1  # Ensure consistency
else:
    raise FileNotFoundError("Tokenizer file not found! Ensure you use the same tokenizer from training.")

# Define sentiment mapping (Ensure it matches training labels)
sentiment_mapping = {0: 'VERY_POSITIVE', 1: 'POSITIVE', 2: 'NEUTRAL', 3: 'NEGATIVE', 4: 'VERY_NEGATIVE', 5: 'NEUTRAL'}

# Function to preprocess images
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found. Using a black placeholder.")
        img_array = np.zeros((224, 224, 3))  # Black placeholder image
    else:
        img = load_img(image_path, target_size=image_size)
        img_array = img_to_array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    return np.expand_dims(img_array, axis=0)  

# Function to preprocess text
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_caption_length)
    return padded_sequence

# Function to preprocess rating
def preprocess_rating(rating):
    return np.array([[rating]])  

# EfficientNet Model for image processing
efficientnet_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3))
efficientnet_model.trainable = False  # Freeze weights

# Define Model Architecture (Ensure same architecture as training)
image_input = Input(shape=(224, 224, 3))
x = efficientnet_model(image_input)
x = GlobalAveragePooling2D()(x)

# Text Model
text_input = Input(shape=(max_caption_length,))
embedding_layer = tf.keras.layers.Embedding(vocab_size, 128)(text_input)
text_features = LSTM(64)(embedding_layer)

# Rating Input
rating_input = Input(shape=(1,))

# Combine Features
combined = Concatenate()([x, text_features, rating_input])
combined = Dense(64, activation='relu')(combined)
output = Dense(6, activation='softmax')(combined)

# Load Model
model = Model(inputs=[image_input, text_input, rating_input], outputs=output)

# Load weights safely
weights_path = r'F:\Multimodal\multimodal_sentiment_weights_six_sentiments2.h5'
if os.path.exists(weights_path):
    try:
        model.load_weights(weights_path)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
else:
    raise FileNotFoundError(f"Weight file '{weights_path}' not found!")

# Function to make predictions
def predict_sentiment(image_path, description, rating):
    image = preprocess_image(image_path)
    text = preprocess_text(description)
    rating = preprocess_rating(rating)

    prediction = model.predict([image, text, rating])
    predicted_label = np.argmax(prediction)  
    sentiment = sentiment_mapping.get(predicted_label, "NEUTRAL")

    print(f"Predicted Sentiment: {sentiment}")
    return sentiment


# Upload view
def upload_sentiment_data(request):
    if request.method == "POST":
        form = SentimentDataForm(request.POST, request.FILES)
        if form.is_valid():
            saved_instance = form.save()
            
            # Get the saved image path
            image_path = saved_instance.image.path  
            text_caption = saved_instance.text  
            description=text_caption
            rating = saved_instance.rating
            
            predicted_sentiment = predict_sentiment(image_path, description, rating)

            return render(request, 'upload_success.html', {'form': form, 'sentiment': predicted_sentiment})  
    else:
        form = SentimentDataForm()

    return render(request, 'upload_form.html', {'form': form})
