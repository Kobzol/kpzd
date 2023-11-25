# Generativní sítě
V této lekci si ukážeme, jak můžeme využít neuronové sítě pro generování či úpravu dat. Konkrétně
si ukážeme [Generativní adversariální sítě (GAN)](https://www.tensorflow.org/tutorials/generative/dcgan).

## Instalace závislosti
```bash
$ pip install tensorflow
```

## Soubory
- [Model](models.py) - definice generátoru a diskriminátoru.
- [Training](train.py) - trénovací kód.
- [Inference](infer.py) - generování MNIST číslic pomocí natrénovaného GANu.
