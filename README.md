# negative-learning

This repository contains the accompanying code for paper "Classification Based on missing features in Deep Convolutional Neural Networks" submitted to NNW journal in December 2018.

# Dependencies

- pytorch (CUDA supported)
- torchvision (for MNIST dataset)
- tqdm

# Running

```
python mnist.py
```

# Example output

```
Testing -- Normal:
[normal] Test set: Average loss: 0.0572, Accuracy: 9824/10000 (98%)
[negative_relu] Test set: Average loss: 0.0633, Accuracy: 9807/10000 (98%)
[hybrid] Test set: Average loss: 0.0464, Accuracy: 9855/10000 (99%)
[hybrid_nr] Test set: Average loss: 0.0716, Accuracy: 9797/10000 (98%)
[hybrid_alt] Test set: Average loss: 0.0638, Accuracy: 9824/10000 (98%)
Testing -- HCUT:
[normal] Test set: Average loss: 1.0740, Accuracy: 6255/10000 (63%)
[negative_relu] Test set: Average loss: 2.5268, Accuracy: 4977/10000 (50%)
[hybrid] Test set: Average loss: 1.2016, Accuracy: 6700/10000 (67%)
[hybrid_nr] Test set: Average loss: 1.1639, Accuracy: 6658/10000 (67%)
[hybrid_alt] Test set: Average loss: 1.1094, Accuracy: 6606/10000 (66%)
Testing -- VCUT:
[normal] Test set: Average loss: 1.6394, Accuracy: 6110/10000 (61%)
[negative_relu] Test set: Average loss: 1.6386, Accuracy: 6324/10000 (63%)
[hybrid] Test set: Average loss: 1.6059, Accuracy: 6884/10000 (69%)
[hybrid_nr] Test set: Average loss: 1.0764, Accuracy: 7015/10000 (70%)
[hybrid_alt] Test set: Average loss: 1.4278, Accuracy: 6578/10000 (66%)
Testing -- DCUT:
[normal] Test set: Average loss: 1.6490, Accuracy: 5495/10000 (55%)
[negative_relu] Test set: Average loss: 2.1363, Accuracy: 5321/10000 (53%)
[hybrid] Test set: Average loss: 1.8628, Accuracy: 5955/10000 (60%)
[hybrid_nr] Test set: Average loss: 1.4446, Accuracy: 6131/10000 (61%)
[hybrid_alt] Test set: Average loss: 1.6316, Accuracy: 6073/10000 (61%)
Testing -- TCUT:
[normal] Test set: Average loss: 2.3135, Accuracy: 3425/10000 (34%)
[negative_relu] Test set: Average loss: 3.0520, Accuracy: 3571/10000 (36%)
[hybrid] Test set: Average loss: 3.1580, Accuracy: 4087/10000 (41%)
[hybrid_nr] Test set: Average loss: 2.4543, Accuracy: 3983/10000 (40%)
[hybrid_alt] Test set: Average loss: 2.2019, Accuracy: 4080/10000 (41%)
```

# Contact

```
nmilosev [at] dmi.rs
```
