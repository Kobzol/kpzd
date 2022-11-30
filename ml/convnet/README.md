# Overfitting
V této lekci si řekneme něco o problému přetrénování (overfitting) a ukážeme si, jak s ním bojovat
(více dat, Dropout vrstva, změna learning rate a batch size).

Dataset Cats vs Dogs lze najít na [Kaggle](https://www.kaggle.com/c/dogs-vs-cats). Stáhněte si jej
a rozbalte do složek `dogs-vs-cats/train` a `dogs-vs-cats/test1`.

## Instalace závislosti
```bash
$ pip install tensorflow tqdm pillow
```

## Soubory
- [Loader](loader.py) - loader pro dataset Cats vs Dogs.
- [Training](train.py) - trénovací kód + definice modelu.

## Užitečné odkazy
- [Vizualizace konvoluce](https://ezyang.github.io/convolution-visualizer/)
- [CS231n - konvoluční sítě](https://cs231n.github.io/convolutional-networks/)
