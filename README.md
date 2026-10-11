# Classificador de Imagens: Gatos vs. Cachorros

Este projeto é um classificador binário de imagens desenvolvido em **Python** utilizando **TensorFlow** e **Keras**. O objetivo é identificar se uma imagem contém um gato ou um cachorro.

Para alcançar uma alta precisão (cerca de **98%**) com pouco tempo de processamento, o projeto utiliza a técnica de **Transfer Learning** (Aprendizado por Transferência) aplicando o modelo pré-treinado **MobileNetV2**.

## Tecnologias Utilizadas

* **TensorFlow & Keras:** Framework principal para a construção e treinamento da rede neural.

* **TensorFlow Datasets (TFDS):** Para o download otimizado e gerenciamento da base de dados oficial `cats_vs_dogs`.

* **API `tf.data`:** Criação de pipelines de processamento de dados em fluxo (*streaming*), otimizando o uso de CPU e GPU.

* **MobileNetV2:** Arquitetura de rede neural convolucional (CNN) pré-treinada no dataset ImageNet, otimizada para dispositivos móveis e visão computacional leve.

* **Matplotlib:** Geração de gráficos e exibição visual das imagens testadas.

* **Requests:** Biblioteca para download seguro de imagens da web para testes avulsos.

## Arquitetura e Pipeline de Dados

### 1. Pipeline de Dados (`tf.data`)

Ao invés de carregar o dataset inteiro na memória RAM, o projeto utiliza um pipeline moderno e eficiente:

* **Format:** As imagens são redimensionadas para `160x160` pixels e seus canais de cor são normalizados de `[0, 255]` para `[-1, 1]`.

* **Batch & Prefetch:** Os dados são divididos em lotes (batches) de 32 imagens. O comando `prefetch(tf.data.AUTOTUNE)` garante que a CPU prepare o próximo lote simultaneamente enquanto a GPU treina o lote atual.

* **Split:** Os dados são divididos em 80% Treinamento, 10% Validação e 10% Teste.

### 2. Transfer Learning (MobileNetV2)

A rede não é treinada "do zero". Nós carregamos a base convolucional do **MobileNetV2** (sem a camada de classificação final) com seus pesos originais treinados na ImageNet.

* **Base Congelada:** Os pesos originais (`~2.2 milhões de parâmetros`) são **congelados** para não perderem o conhecimento prévio sobre bordas, formas e texturas.

* **Classification Head:** Adicionamos uma camada `GlobalAveragePooling2D` e uma camada densa simples (`Dense(1)`) com apenas **1.281 parâmetros treináveis**. O modelo só precisa aprender a diferenciar as "features" genéricas extraídas em duas classes: gato (0) ou cachorro (1).

## Como Executar o Projeto

Este projeto foi desenhado para rodar perfeitamente no **Google Colab**.

### Passo 1: Preparação do Ambiente

Abra um novo notebook no Google Colab e certifique-se de ativar a Aceleração por GPU (`Ambiente de Execução > Alterar tipo de ambiente de execução > GPU T4`).

Em seguida, instale a dependência de pacotes necessária executando o comando abaixo na primeira célula:

```
%pip install importlib_resources

```

*(Nota: Reinicie a sessão após a instalação, se solicitado pelo Colab).*

### Passo 2: Execução das Células

O projeto está modularizado nas seguintes etapas:

1. **Célula 1 (Setup e Dados):** Importa as bibliotecas, faz o download do dataset de +23 mil imagens via TFDS e monta o pipeline `tf.data`.

2. **Célula 2 (Modelo):** Carrega o MobileNetV2, congela suas camadas, adiciona as novas camadas de predição e compila o modelo usando o otimizador RMSprop e `BinaryCrossentropy`.

3. **Célula 3 (Treinamento):** Treina o modelo em 5 épocas (`epochs`) iniciais e exibe a acurácia no conjunto de teste.

4. **Célula 4 (Teste da Web):** Permite inserir uma URL (link) de qualquer imagem da internet para testar o classificador na prática.

## Testando Suas Próprias Imagens

A última célula do notebook permite testar qualquer imagem da internet. A função implementada automaticamente baixa a imagem, processa, aplica o modelo e exibe a imagem na tela com a probabilidade prevista (Gato ou Cachorro).

**Basta alterar a variável `url_teste` no código:**

```
url_teste = "COLOQUE_SEU_LINK_AQUI.jpg"
testar_imagem_da_web(url_teste)
```
