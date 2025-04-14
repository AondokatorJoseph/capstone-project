import tensorflow as tf
from tensorflow.keras import layers, models

class WasteClassifier:
    def __init__(self, num_classes=6, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.category_mapping = {
            0: 'glass',
            1: 'paper',
            2: 'cardboard',
            3: 'plastic',
            4: 'metal',
            5: 'trash'
        }

    def build_model(self):
        """Build CNN model optimized for TrashNet dataset"""
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        
        # Fine-tune the last few layers
        base_model.trainable = True
        for layer in base_model.layers[:-30]: # train the first 30 layers
            layer.trainable = False

        
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Use learning rate schedule instead of fixed rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,  # Higher initial rate
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model

    def train(self, train_data, validation_data, epochs=20, batch_size=32):
        """Train the model on TrashNet data"""
        if self.model is None:
            self.build_model()
            

        # Apply augmentation to training data
        train_images, train_labels = train_data
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Create validation dataset
        val_ds = tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)
            
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        return history

    def evaluate_on_coco(self, coco_data):
        """Evaluate model trained on TrashNet using COCO dataset"""
        test_images, test_labels = coco_data
        results = self.model.evaluate(test_images, test_labels, verbose=1)
        return {
            'test_loss': results[0],
            'test_accuracy': results[1]
        }