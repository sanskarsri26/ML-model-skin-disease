import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 64  # Adjust batch size if memory allows

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.9, 1.1],
    validation_split=0.2  # Split data into training and validation sets
)

# Train and validation generators
train_generator = train_datagen.flow_from_directory(
    '/kaggle/input/skin-diseases-image-dataset/IMG_CLASSES',  # Update with your dataset path
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    '/kaggle/input/skin-diseases-image-dataset/IMG_CLASSES',  # Update with your dataset path
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Use MobileNetV2 as a lightweight alternative to VGG16
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_width, img_height, 3),
    include_top=False,
    weights='/kaggle/input/mobinet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
)

base_model.trainable = False  # Freeze the base model initially

# Define the model architecture
model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(train_generator.num_classes, activation='softmax', dtype='float32')  # Adjust to number of classes
])

# Compile the model with a lower learning rate for initial training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# Callbacks for early stopping, reducing learning rate, and saving best model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
checkpoint = ModelCheckpoint(
    'best_model.keras', 
    monitor='val_loss', 
    save_best_only=True, 
    save_weights_only=False, 
    mode='min'
)

# Train the model with callbacks
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,  # Training for 10 epochs
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# Save the trained model
model.save('/kaggle/working/skin_disease_model_mobilenet.h5')
