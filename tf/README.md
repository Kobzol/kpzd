# Tensorflow
V této lekci si řekneme něco o knihovně [`TensorFlow`](https://www.tensorflow.org/), která slouží pro
vysoce efektivní práci s tensory (vícedimenzionálními poli). Umožňuje nám napsat výpočet v relativně
přehledném Python kódu, a pak ho zakompilovat do optimalizovaného výpočetního grafu, který je schopen
běžet na akcelerátoru (např. GPU). Zároveň nám umožňuje automaticky spočítat derivaci tohoto kódu,
což se hodí pro neuronové sítě.

## Instalace závislosti
```bash
$ pip install tensorflow
```

## Notebooky
- [Základy TF](tf-basics.ipynb) - základy `TensorFlow`, práce s tensory. 
- [Kompilace](tf-compilation.ipynb) - zakompilování Python funkce do `TensorFlow` grafu.
- [Automatická derivace](tf-autodiff.ipynb) - využití `GradientTape` k derivaci TF výpočtu.
- [Fitování křivky](tf-training.ipynb) - využití automatické derivace k nafitování křivky polem
2D bodů.
