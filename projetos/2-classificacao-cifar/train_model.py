import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 2 — Classificação CIFAR-10
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset CIFAR-10 via tf.keras.datasets.cifar10
#   2. Normalizar as imagens para [0, 1] (shape (32, 32, 3))
#   3. Separar um conjunto de validação
#   4. Incluir data augmentation (ex: layers.RandomFlip, RandomRotation, RandomZoom)
#      aplicada ao conjunto de treino
#   5. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   6. Treinar com EarlyStopping monitorando a perda de validação
#   7. Exibir a acurácia de validação final no terminal
#   8. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

#=
# Etapa 1 — Treinamento do Modelo (train_model.py)
#=
RNG_SEED = 25
VAL_FRACTION = 0.1
BATCH_SIZE = 64
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

tf.random.set_seed(RNG_SEED)

#=
# 1.a. Carregamento do dataset CIFAR-10 via TensorFlow
#=
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train_full = y_train_full.squeeze(axis=-1)
y_test = y_test.squeeze(axis=-1)

print(f"x_train_full.shape: {x_train_full.shape}")
print(f"y_train_full.shape: {y_train_full.shape}")

#=
# 1.b. Split explícito treino/validação
#=
n_total = x_train_full.shape[0]
rng = tf.random.Generator.from_seed(RNG_SEED)
permut = tf.random.shuffle(tf.range(n_total), seed=RNG_SEED).numpy()

n_val = int(n_total * VAL_FRACTION)
val_idx, train_idx = permut[:n_val], permut[n_val:]

x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]
x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]

print(f"Training: {x_train.shape[0]} samples | Validation: {x_val.shape[0]} samples | Testing: {x_test.shape[0]} samples")

#=
# 1.c. Data augmentation aplicada ao conjunto de treino, usando camadas do Keras (ex: RandomFlip("horizontal"), RandomRotation, RandomZoom) incorporadas ao modelo ou ao pipeline de treino
#=
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.08, 0.08),
    ],
    name="data_augmentation",
)

#=
# 1.d. Construção de uma CNN com 3-4 blocos convolucionais (Conv2D + BatchNormalization MaxPooling2D) seguida de Dropout
#=
def conv_block(x, filters, block_id):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{block_id}_a")(x)
    x = layers.BatchNormalization(name=f"bn{block_id}_a")(x)
    x = layers.Activation("relu", name=f"relu{block_id}_a")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{block_id}_b")(x)
    x = layers.BatchNormalization(name=f"bn{block_id}_b")(x)
    x = layers.Activation("relu", name=f"relu{block_id}_b")(x)
    x = layers.MaxPooling2D(pool_size=2, name=f"pool{block_id}")(x)

    return x


inputs = keras.Input(shape=(32, 32, 3), name="input_image")
x = data_augmentation(inputs)

x = conv_block(x, 32, 1)
x = conv_block(x, 64, 2)
x = conv_block(x, 128, 3)
x = conv_block(x, 256, 4)

x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.5, name="dropout")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs, outputs, name="cifar10_cnn")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

#=
# 1.e. Treinamento com early stopping baseado na perda de validação
#=
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-5,
    verbose=1,
)

history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=MAX_EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=2,
)

#=
# 1.f. Exibição da acurácia de validação final no terminal
#=
val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("#=")
print(f"# Training epochs: {len(history.history['loss'])}")
print(f"# Final validation accuracy: {val_accuracy:.4f} (loss: {val_loss:.4f})")
print(f"# Final testing accuracy: {test_accuracy:.4f} (loss: {test_loss:.4f})")
print("#=")

#=
# 1.g. Salvamento do modelo treinado em formato Keras (model.h5)
#=
model.save("model.h5")
