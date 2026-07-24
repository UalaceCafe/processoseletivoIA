import tensorflow as tf
import keras
import os

# ---------------------------------------------------------------------------
# Projeto 2 — Otimização do Modelo (CIFAR-10)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

#=
# Etapa 2 — Otimização do Modelo (optimize_model.py)
#=
MODEL_FILE = "model.h5"
TFLITE_FILE = "model.tflite"

#=
# 2.a. Carregamento do model.h5 treinado
#=
model = keras.models.load_model(MODEL_FILE)
model.summary()

#=
# 2.b. Conversão para TensorFlow Lite (model.tflite)
# 2.c. Aplicação de uma técnica de otimização (ex: Dynamic Range Quantization)
#=
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

#=
with open(TFLITE_FILE, "wb") as f:
    f.write(tflite_model)

original_size = os.path.getsize(MODEL_FILE)
tflite_size = os.path.getsize(TFLITE_FILE)

print(f"Original size ({MODEL_FILE}):  {original_size / 1024:.1f} KB")
print(f"Optimized size ({TFLITE_FILE}): {tflite_size / 1024:.1f} KB")
print(f"Reduction: {100 * (1 - tflite_size / original_size):.1f}%")
