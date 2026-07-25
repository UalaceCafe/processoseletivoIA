# Projeto 2 — Classificação CIFAR-10

## 📝 Relatório do Candidato

👤 **Nome Completo:** Ualace Henrique Santos Café

### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura implementada recebe imagens `(32, 32, 3)` normalizadas para `[0, 1]` e é composta por quatro blocos convolucionais com número crescente de filtros (32 -> 64 -> 128 -> 256). Cada bloco contém duas camadas `Conv2D` (kernel 3x3, sem bias, já que, dado a `BatchNormalization`, o bias é redundante) seguidas de `BatchNormalization` + `ReLU`, terminando em um `MaxPooling2D`. Após o quarto bloco, uma `GlobalAveragePooling2D` substitui o método comum de usar uma camada de `Flatten` + uma `Dense` grande (isso foi feito para reduzir consideravelmente o número de parâmetros e o risco de overfitting, o que é especialmente relevante em um dataset pequeno como o CIFAR-10 treinado com poucas épocas em CPU). Um `Dropout(0.5)` antecede a camada final `Dense(10, activation="softmax")`.

A estratégia de data augmentation (`RandomFlip("horizontal")`, `RandomRotation`, `RandomZoom` e `RandomTranslation`) foi incorporada como camadas Keras dentro do próprio grafo do modelo, logo após a entrada. Isso tem duas vantagens práticas: as transformações ficam automaticamente ativas apenas durante o fitting do modelo (`model.fit()`) e inertes durante a `evaluate()`/inferência, sem necessidade de pipelines separados, e assim a augmentation passa a fazer parte do artefato salvo.

### 2️⃣ Bibliotecas Utilizadas

- **TensorFlow / Keras** — `tensorflow==2.15.1` (`keras==2.15.0`), usada para construção, treinamento, avaliação e salvamento do modelo (`train_model.py`), além da conversão para TensorFlow Lite (`optimize_model.py`) e da inferência com o interpretador (`run_inference.py`).
- **NumPy** — manipulação de arrays e amostragem em `run_inference.py`.
- **Bibliotecas padrão do Python** — `os`, `sys` e `io`, usadas para localizar caminhos de arquivo de forma independente do diretório de execução, normalizar a codificação de saída do terminal.

### 3️⃣ Técnica de Otimização do Modelo

Em `optimize_model.py`, o modelo é convertido para TensorFlow Lite via o método `tf.lite.TFLiteConverter.from_keras_model()` com `converter.optimizations = [tf.lite.Optimize.DEFAULT]`, o que aplica **Dynamic Range Quantization** - ou seja, os pesos são quantizados de float32 para int8, enquanto as ativações permanecem em float32 e são re-escaladas em tempo de execução. Essa técnica foi escolhida em vez da quantização full-integer porque não exige um `representative_dataset` de calibração — reduzindo o risco de erro em um pipeline que precisa rodar sem intervenção manual, mas  ainda assim entregando uma redução significativa de tamanho do artefato final.

### 4️⃣ Resultados Obtidos

- **Acurácia de validação final:** 0.8472 (loss: 0.4578)
- **Acurácia de teste final:** 0.8462 (loss: 0.4634)
- **Tamanho de `model.h5`:** 13944.5 KB / 13.6 MB
- **Tamanho de `model.tflite`:** 1169.7 KB / 1.14 MB (redução de 91.6%)

### 5️⃣ Comentários Adicionais

A principal dificuldade não foi de modelagem, mas de compatibilidade entre ambientes: o `model.h5` treinado localmente embute nos configs dos inicializadores `VarianceScaling` (como `GlorotUniform`) os parâmetros `input_axes`/`output_axes`, introduzidos em uma versão recente do Keras. O ambiente de CI usado para validação roda uma versão de Keras mais antiga, cujo `GlorotUniform.__init__()` não reconhece esses argumentos. Assim, o carregamento falhava com `TypeError: GlorotUniform.__init__() got an unexpected keyword argument 'input_axes'` logo na desserialização da primeira camada convolucional. Para contornar isto, a solução adotada foi fixar uma versão do Tensorflow (`tensorflow==2.15.1`, mais especificamente) nos requerimentos, ao invés do original `tensorflow>=2.12`. Desse modo, foi possível garantir que ambos o workspace local quanto o do CI estivessem rodando a mesma versão dos pacotes requeridos.

Outra decisão técnica relevante foi restringir o treinamento a `EarlyStopping` (monitorando `val_loss`, com `restore_best_weights=True`) combinado a `ReduceLROnPlateau`, dado o orçamento limitado de épocas/CPU imposto pelas restrições do projeto. Essa combinação permitiu extrair mais desempenho da rede antes do critério de parada ser acionado, sem exigir treinar por mais épocas do que o necessário.

### 6️⃣ Exemplo de Inferência

Rodando inferência (`run_inference.py`) em 5 amostras usando o modelo `model.tflite`:

```
Amostra 1: predito=cat | real=cat
Amostra 2: predito=ship | real=ship
Amostra 3: predito=ship | real=ship
Amostra 4: predito=airplane | real=airplane
Amostra 5: predito=frog | real=frog
```

Todas as 5 amostras testadas tiveram a classe predita coincidindo com a classe real, sem nenhum erro de classificação nesse lote. Isso é consistente com a acurácia relativamente alta de validação obtida (0.8472), mas vale notar que 5 amostras é uma base pequena demais para tirar conclusões - um lote maior provavelmente revelaria confusões entre classes visualmente próximas (ex: gato/cão, automóvel/caminhão), como é típico em bases do tipo.
