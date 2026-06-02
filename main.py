# -*- coding: utf-8 import os
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import clean_and_split_data, WORK_DIR, CLASSES, SEED

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def get_generators(rotation_range=20, use_brightness=True):
    aug_args = {
        'rescale': 1./255,
        'rotation_range': rotation_range,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'horizontal_flip': True,
        'fill_mode': 'nearest'
    }
    if use_brightness:
        aug_args['brightness_range'] = [0.8, 1.2]
    
    train_datagen = ImageDataGenerator(**aug_args)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(WORK_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=SEED,
        shuffle=True
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(WORK_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=SEED,
        shuffle=False
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(WORK_DIR, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=SEED,
        shuffle=False
    )
    return train_gen, val_gen, test_gen

def build_model():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ])
    return model

def train_model(rotation_range=20, learning_rate=0.001, epochs=50, experiment_name='exp'):
    use_brightness = (rotation_range > 0)
    train_gen, val_gen, test_gen = get_generators(
        rotation_range=rotation_range,
        use_brightness=use_brightness
    )
    
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    print(f'\n>>> 实验 [{experiment_name}]')
    print(f'rotation_range={rotation_range}, learning_rate={learning_rate}, epochs={epochs}')
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save(f'model_{experiment_name}.h5')
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.savefig(f'curve_{experiment_name}.png')
    plt.close()
    
    best_val_acc = max(history.history['val_accuracy'])
    print(f'最佳验证准确率：{best_val_acc:.4f}')
    return model, test_gen

def final_evaluate(model, test_gen, experiment_name='final'):
    loss, accuracy = model.evaluate(test_gen, verbose=0)
    print(f'\n========== 最终测试 [{experiment_name}] ==========')
    print(f'测试集 Loss：{loss:.4f}')
    print(f'测试集 Accuracy：{accuracy:.4f}')
    
    y_pred = np.argmax(model.predict(test_gen, verbose=0), axis=1)
    y_true = test_gen.classes
    
    print('\n分类报告：')
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{experiment_name}.png')
    plt.close()
    return accuracy

if __name__ == '__main__':
    # clean_and_split_data()
    
    model, test_gen = train_model(
        rotation_range=0,
        learning_rate=0.001,
        epochs=50,
        experiment_name='baseline'
    )
    
    # final_evaluate(model, test_gen, experiment_name='final')