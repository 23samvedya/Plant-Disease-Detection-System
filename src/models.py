"""Transfer-learning CNNs with preprocessing built into the model.

Putting each backbone's own `preprocess_input` inside the model means a saved
model takes raw RGB pixels in [0, 255]. The app (and anyone else) only has to
resize the image, so training and inference cannot drift apart again. This
fixes two bugs in the 2023 code:
  * ResNet50 was trained on raw pixels but app.py applied resnet.preprocess_input
    before predicting, so the deployed model saw differently scaled inputs.
  * The InceptionV3 notebook defined ImageDataGenerator(rescale=1/252) but never
    used it, so InceptionV3 received raw 0-255 pixels instead of the [-1, 1]
    range it was trained on.
"""
import keras
from keras import layers

from src.config import CLASS_NAMES, IMG_SIZE

BACKBONES = {
    "resnet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input),
    "inceptionv3": (keras.applications.InceptionV3, keras.applications.inception_v3.preprocess_input),
}


@keras.saving.register_keras_serializable(package="fruit")
class BackbonePreprocess(layers.Layer):
    """Applies the named backbone's preprocess_input. Serializable, so it is saved with the model."""

    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self._fn = BACKBONES[backbone][1]

    def call(self, x):
        return self._fn(x)

    def get_config(self):
        return {**super().get_config(), "backbone": self.backbone}


def build_cnn(backbone, dense_units=512, dropout=0.3, augment=True, preprocess=True):
    """`preprocess=False` reproduces the 2023 bug (raw pixels into the backbone); ablation only."""
    base_cls, _ = BACKBONES[backbone]
    base = base_cls(include_top=False, weights="imagenet",
                    input_shape=(*IMG_SIZE, 3), pooling="avg")
    base.trainable = False  # same frozen-backbone setup as the 2023 project

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="rgb_0_255")
    x = inputs
    if augment:  # active only during training
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
    if preprocess:
        x = BackbonePreprocess(backbone)(x)
    x = base(x, training=False)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_fruit")
