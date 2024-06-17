import os
from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
import numpy as np
import tensorflow as tf
import keras

# Load your trained model with custom objects
model_path = ('/Users/user/Desktop/final_year/final_year_project/models/vgg16_aqi_model.h5')
model = keras.models.load_model(model_path, custom_objects={'mse': keras.losses.MeanSquaredError()})

def handle_uploaded_file(f):
    file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', f.name)
    with open(file_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return file_path

def predict_aqi(img_path):
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.vgg16.preprocess_input(img_array)

    prediction = model.predict(img_array)
    return prediction[0][0]

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_instance = form.save()
            img_path = handle_uploaded_file(request.FILES['image'])
            aqi = predict_aqi(img_path)
            return render(request, 'result.html', {'aqi': aqi})
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})
