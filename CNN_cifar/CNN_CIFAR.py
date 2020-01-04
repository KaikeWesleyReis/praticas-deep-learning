#!/usr/bin/env python
# coding: utf-8

### Classicação do CIFAR-10 usando Convolucional Neural Networks
# Autor: Kaike
# OBS: Para melhor entendimento veja o notebook referente a esse código

### Módulos utilizados
# Biblioteca básica do python
import numpy as np
# Para o desenvolvimento dos modelos de Deep Learning
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import datasets, models, layers, optimizers
# Para fazer alguns gráficos caso necessário
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Para aplicar a otimização de parametros da rede
import talos


### CIFAR-10
# Importando banco de dados
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data();
# Normalizando os valores dos pixels para o intervalo de 0 a 1
train_images, test_images = train_images/255.0, test_images/255.0
# Definindo as classes
class_names = ['airplane','automobile','bird','cat', 'deer','dog','frog','horse','ship','truck']
# Analisando algumas imagens do CIFAR
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
# Para este notebook irei juntar ambos os conjuntos train/test (por causa do otimizador talos)
x = np.concatenate((train_images, test_images))
y = np.concatenate((train_labels, test_labels))


### Parte 1: Criando a Rede de Convolução (CNN)
# Definindo os dimensões de entrada para o modelo
image_shape = (32,32,3)
# Declarando objeto
cnn = models.Sequential()
# Definindo a primeira camada, note que aqui o formato da imagem é aplicado
cnn.add(layers.Conv2D(filters=32, 
                      kernel_size=(3,3),
                      strides=(1,1), 
                      padding='same',
                      activation='relu', 
                      input_shape=image_shape,
                      name='conv2D_input'))
# Definindo a segunda camada
cnn.add(layers.MaxPool2D(pool_size=(2,2),
                         name='maxPool_1'))
# Definindo a terceira camada
cnn.add(layers.Conv2D(filters=64,
                      kernel_size=(3,3),
                      strides=(1,1),
                      padding='same',
                      activation='relu',
                      name='conv2D_1'))
# Definindo a quarta camada
cnn.add(layers.MaxPool2D(pool_size=(2,2),
                         name='maxPool_2'))
# Definindo a quinta camada
cnn.add(layers.Conv2D(filters=128,
                      kernel_size=(3,3),
                      strides=(1,1),
                      padding='same',
                      activation='relu',
                      name='conv2D_2'))
# Definindo a sexta camada
cnn.add(layers.MaxPool2D(pool_size=(2,2),
                         name='maxPool_3'))

### Parte 2: Criando a Rede de Toda-Conectada (DNN)
# Flatten N-D Vetor em 1-D Vetor
cnn.add(layers.Flatten())
# Primeira camada para DNN
cnn.add(layers.Dense(units = 128,
                     activation='relu',
                     name='dense_1'))
# Saída
cnn.add(layers.Dense(units = 10,
                     activation='softmax',
                     name='dense_output'))


### Parte 3: Compilando o modelo
cnn.compile(optimizer='Adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


### Parte 4: Treinando o modelo
cnn_training = cnn.fit(x = x,
                       y = y,
                       epochs = 5,
                       batch_size=32,
                       validation_split=0.3,
                       #validation_data=(test_images, test_labels),
                       verbose=2                  
                      )


## Avaliando a performance durante o treinamento do modelo
epochs_range = range(0,5)
# Plots
fig, ax = plt.subplots(1,2, figsize=(15,5));
# Accuracy X Epochs
ax[0].plot(cnn_training.history['accuracy'], label='Train Accuracy')
ax[0].plot(cnn_training.history['val_accuracy'], label = 'Validation accuracy')
ax[0].set_title('CNN - Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].set_ylim([0, 1])
ax[0].set_xticks(epochs_range)
ax[0].legend(loc='lower right')
# MSE X Epochs
ax[1].plot(cnn_training.history['loss'], label='Train Loss')
ax[1].plot(cnn_training.history['val_loss'], label = 'Validation Loss')
ax[1].set_title('CNN - Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].set_xticks(epochs_range)
ax[1].legend(loc='upper right')
# Default to improve spaces between the plots
plt.tight_layout()


### Parte 5: Melhorando o modelo
# Passo 1 - Definindo o dict de parametros
params = {
          'conv_layers':[1,2,4],
          'dense_hidden_layers':[2,3,4],
         }
# Passo 2 - Definindo uma função para geração do modelo keras
def cnn_cifar_model(x_train, y_train, x_val, y_val, parameters):
    # Definindo a lista da quantidade de filtros/camadas
    n_filters = [32,64,128,256]
    n_units = [128,256,512,1024]
    
    # Iniciando o model assim como foi feito anteriormente
    CNN = models.Sequential()
    # Desenvolvimento da CNN - Geração automática das camadas
    for conv, filtros in zip(range(0, parameters['conv_layers']), n_filters):
        # Primeira camada precisa das dimensões da imagem
        if conv == 0:
            ## 1 camada de Convolucao
            NAME = 'conv2D_input'
            CNN.add(layers.Conv2D(filters = filtros, 
                                  kernel_size = (3,3),
                                  strides = (1,1), 
                                  padding = 'same',
                                  activation = 'relu', 
                                  input_shape = image_shape,
                                  name = NAME))
            ## 1 camada de Max Pool
            NAME = 'maxPool_{}'.format(conv)
            CNN.add(layers.MaxPool2D(pool_size=(2,2),
                                     name=NAME))
        
        # Outras camadas
        else:
            ### Adicionar camada de Convolucao
            NAME = 'conv2D_{}'.format(conv)
            CNN.add(layers.Conv2D(filters=filtros, 
                                  kernel_size=(3,3),
                                  strides=(1,1), 
                                  padding='same',
                                  activation='relu', 
                                  name=NAME))
            ### Adicionar camada de Max Pool
            NAME = 'maxPool_{}'.format(conv + 1)
            CNN.add(layers.MaxPool2D(pool_size=(2,2),
                                     name=NAME))              

    # Desenvolvimento da camada FLATTEN
    CNN.add(layers.Flatten())
    
    # Desenvolvimento da DNN - Geração automática das camadas
    for dense, neuronios in zip(range(0, parameters['dense_hidden_layers']), n_units):
        # Gerar camada
        NAME = 'dense_{}'.format(dense)
        CNN.add(layers.Dense(units = neuronios,
                             activation='relu',
                             name=NAME))   

    # Adicionar camada de saída
    CNN.add(layers.Dense(units = 10,
                         activation='softmax',
                         name='dense_output'))  
    
    # Compilando o modelo
    CNN.compile(optimizer='Adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    # Treinando o modelo
    CNN_training = CNN.fit(x = x_train,
                           y = y_train,
                           epochs = 5,
                           validation_data = (x_val, y_val),
                           verbose = 0)
    
    return CNN_training, CNN
# Passo 3 - Fazer a avaliação
analise = talos.Scan(x = x,
                     y = y,
                     params = params,
                     model = cnn_cifar_model,
                     experiment_name = 'cnn')


### Parte 6: Obtivemos sucesso com a otimização?
# variavel auxiliar
df = analise.data
# criando coluna categorica para melhor visualizacao
for i in range(0,len(df)):
    df.loc[i,'Camadas Testadas'] = "Conv: {} | Dense:{}".format(df.loc[i,'conv_layers'],df.loc[i,'dense_hidden_layers']) 
# Define tamanho da imagem
plt.figure(figsize=(15,10))
# Plot
sns.scatterplot(x="val_loss", y="val_accuracy", hue='Camadas Testadas', data=df, s=100,cmap='dark');
# Colocar legenda fora do grafico
plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), ncol=1);
# Labels
plt.title('Loss VS Accuracy');
plt.xlabel('Loss Validação');
plt.ylabel('Accuracy Validação');

## RESULTADO: Melhor modelo ainda continua o modelo original
