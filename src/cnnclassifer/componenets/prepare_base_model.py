import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from cnnclassifer.config.configuration import ConfigurationManager
from cnnclassifer.entity.config_entity import PrepareBaseModelConfig
from cnnclassifer import logger
from pathlib import Path

class PrepareBaseModel:
    def __init__(self,config=PrepareBaseModelConfig):
        self.config=config
    
    def get_base_model(self):
        self.model = tf.keras.applications.EfficientNetB0(
        input_shape=self.config.params_image_size,
        weights=self.config.params_weights,      # usually "imagenet"
        include_top=self.config.params_include_top  # False if you’re adding your own classifier
    )

        self.save_model(path=str(self.config.base_model_path),model=self.model)

    @staticmethod
    def save_model(path: str, model: tf.keras.Model):
        # Solution 1: Use SavedModel format (recommended)
        try:
            model.save(path, save_format='tf')
        except Exception as e:
            print(f"SavedModel format failed: {e}")
            # Solution 2: Use HDF5 format as fallback
            try:
                if not path.endswith('.h5'):
                    path = path + '.h5'
                model.save(path, save_format='h5')
            except Exception as e2:
                print(f"HDF5 format also failed: {e2}")
                # Solution 3: Save weights only
                model.save_weights(path + '_weights.h5')
                print(f"Saved weights only to {path}_weights.h5")

    @staticmethod    
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        prediction = tf.keras.layers.Dense(units=classes, activation='softmax')(x)


        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )


        full_model.summary()
        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=str(self.config.updated_base_model_path), model=self.full_model)
    

    

if __name__=="__main__":
    try:
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()  # Fixed method name
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
    except Exception as e:
        raise e