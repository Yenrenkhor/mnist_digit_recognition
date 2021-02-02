from keras.models import model_from_json
from keras.initializers import glorot_uniform
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def init():
    print(os.getcwd())
    model = load_model('model', compile=True)
    model._make_predict_function()
    print("Loaded Model from disk")

    # Compile and evaluate loaded model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph = tf.Graph()

    return model, graph


