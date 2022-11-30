import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras import backend as K

from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils import num_of_gpus
from tf_keras_vis.saliency import Saliency

from tensorflow.keras.applications.vgg16 import VGG16 as Model

model = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)
# model.summary()

replace2linear = ReplaceToLinear()

score = CategoricalScore([272, 90, 60])

# Image titles
image_titles = ['coyote', 'lorikeet', 'night_snake']

# Load images and Convert them to a Numpy array
img1 = load_img('images/coyote.jpeg', target_size=(224, 224))
img2 = load_img('images/lorikeet.jpeg', target_size=(224, 224))
img3 = load_img('images/night_snake.jpeg', target_size=(224, 224))
images = np.asarray([np.array(img1), np.array(img2), np.array(img3)])

# Preparing input data for VGG16
X = preprocess_input(images)

# Rendering
f, ax = plt.subplots(nrows=1, ncols=1)
for i, title in enumerate(image_titles):
    # ax[i].set_title(title, fontsize=16)
    ax.imshow(images[i])
    ax.axis('off')
    plt.tight_layout()
    name = title + ".jpg"
    plt.savefig(name,bbox_inches='tight')


def gradcam():
    gradcam = Gradcam(model, model_modifier=replace2linear, clone=True)
    cam = gradcam(score, X, penultimate_layer=-1)

    f, ax = plt.subplots(nrows=1, ncols=1)
    for i, title in enumerate(image_titles):
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        # ax[i].set_title(title, fontsize=16)
        ax.imshow(images[i])
        ax.imshow(heatmap, cmap='jet', alpha=0.5) # overlay
        ax.axis('off')
        plt.tight_layout()
        name = "output/gradcam" + title + ".jpg"
        plt.savefig(name,bbox_inches='tight')

def gradcamplusplus():
    gradcam = GradcamPlusPlus(model, model_modifier=replace2linear, clone=True)
    cam = gradcam(score, X, penultimate_layer=-1)

    f, ax = plt.subplots(nrows=1, ncols=1)
    for i, title in enumerate(image_titles):
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        ax.set_title(title, fontsize=16)
        ax.imshow(images[i])
        ax.imshow(heatmap, cmap='jet', alpha=0.5)
        ax.axis('off')
        plt.tight_layout()
        name = "output/gradcam_plus_plus" + title + ".jpg"
        plt.savefig(name)

def scorecam():
    scorecam = Scorecam(model)
    cam = scorecam(score, X, penultimate_layer=-1, max_N=10)

    f, ax = plt.subplots(nrows=1, ncols=1)
    for i, title in enumerate(image_titles):
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        # ax[i].set_title(title, fontsize=16)
        ax.imshow(images[i])
        ax.imshow(heatmap, cmap='jet', alpha=0.5)
        ax.axis('off')
        plt.tight_layout()
        name = "output/scorecam" + title + ".jpg"
        plt.savefig(name,bbox_inches='tight')

def smoothgrad():
    saliency = Saliency(model, model_modifier=replace2linear, clone=True)
    saliency_map = saliency(score, X, smooth_samples=20,smooth_noise=0.20)

    f, ax = plt.subplots(nrows=1, ncols=1)
    for i, title in enumerate(image_titles):
        # ax[i].set_title(title, fontsize=16)
        ax.imshow(saliency_map[i], cmap='gray')
        ax.axis('off')
        plt.tight_layout()
        name = "output/smoothgrad" + title + ".jpg"
        plt.savefig(name,bbox_inches='tight')

# gradcam()
# smoothgrad()
# gradcamplusplus()
scorecam()