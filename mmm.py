import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping

# Set directory paths
train_dir = '/Users/manasikonidala/Downloads/horse-or-human/horse-or-human/train'
test_dir = '/Users/manasikonidala/Downloads/horse-or-human/horse-or-human/validation'

# 1. Load the dataset using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# 2. Visualize some samples from the dataset
sample_images, sample_labels = next(train_generator)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(sample_images[i])
    plt.title(f'Label: {sample_labels[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# 3. Build your own CNN model for binary classification
def build_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),  # Prevent overfitting
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 4. Train the model with early stopping to prevent overfitting
cnn_model = build_cnn_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = cnn_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    callbacks=[early_stopping]
)

# 5. Evaluate the model on the test set
test_loss, test_acc = cnn_model.evaluate(test_generator)
print(f'Test accuracy: {test_acc * 100:.2f}%')

# 6. Confusion Matrix for CNN Model
cnn_predictions = cnn_model.predict(test_generator)
cnn_predictions = (cnn_predictions > 0.5).astype('int32')

cnn_cm = confusion_matrix(test_generator.classes, cnn_predictions)
print('CNN Model Confusion Matrix:')
print(cnn_cm)

# 7. Compare with Pre-trained Models: VGG16 and ResNet50

def build_pretrained_model(base_model):
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load VGG16 and ResNet50 with pre-trained weights
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
resnet50_base = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the convolutional layers of both models
vgg16_base.trainable = False
resnet50_base.trainable = False

# Build and compile the models
vgg16_model = build_pretrained_model(vgg16_base)
resnet50_model = build_pretrained_model(resnet50_base)

# 8. Train Pre-trained Models
history_vgg16 = vgg16_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

history_resnet50 = resnet50_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

# 9. Evaluate Pre-trained Models on Test Data
vgg16_test_loss, vgg16_test_acc = vgg16_model.evaluate(test_generator)
resnet50_test_loss, resnet50_test_acc = resnet50_model.evaluate(test_generator)

print(f'VGG16 Test accuracy: {vgg16_test_acc * 100:.2f}%')
print(f'ResNet50 Test accuracy: {resnet50_test_acc * 100:.2f}%')

# 10. Confusion Matrices for Pre-trained Models
vgg16_predictions = vgg16_model.predict(test_generator)
resnet50_predictions = resnet50_model.predict(test_generator)

vgg16_predictions = (vgg16_predictions > 0.5).astype('int32')
resnet50_predictions = (resnet50_predictions > 0.5).astype('int32')

vgg16_cm = confusion_matrix(test_generator.classes, vgg16_predictions)
resnet50_cm = confusion_matrix(test_generator.classes, resnet50_predictions)

print('VGG16 Confusion Matrix:')
print(vgg16_cm)
print('ResNet50 Confusion Matrix:')
print(resnet50_cm)

# 11. Plot the Training History of Models
def plot_history(history, model_name):
    plt.plot(history.history['accuracy'], label=f'{model_name} Train Accuracy')
    plt.plot(history.history['val_accuracy'], label=f'{model_name} Val Accuracy')
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Plot histories for CNN, VGG16, and ResNet50
plot_history(history, 'CNN')
plot_history(history_vgg16, 'VGG16')
plot_history(history_resnet50, 'ResNet50')
