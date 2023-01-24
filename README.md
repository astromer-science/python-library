# ASTROMER Python library 🔭

ASTROMER is a transformer based model pretrained on millions of light curves. ASTROMER can be finetuned on specific datasets to create useful representations that can improve the performance of novel deep learning models.

❗ This version of ASTROMER can only works on single band light curves.

🔥 [See the official repo here](https://github.com/astromer-science/main-code)

## Install
```
pip install ASTROMER
```

## How to use it
Currently, there are 2 pre-trained models: `macho` and `atlas`.
To load weights use:
```
from ASTROMER.models import SingleBandEncoder

model = SingleBandEncoder()
model = model.from_pretraining('macho')
```
It will automatically download the weights from [this public github repository](https://github.com/astromer-science/weights.git) and load them into the `SingleBandEncoder` instance.

Assuming you have a list of vary-lenght (numpy) light curves.
```
import numpy as np

samples_collection = [ np.array([[5200, 0.3, 0.2],
                                 [5300, 0.5, 0.1],
                                 [5400, 0.2, 0.3]]),

                       np.array([[4200, 0.3, 0.1],
                                 [4300, 0.6, 0.3]]) ]

```
Light curves are `Lx3` matrices with time, magnitude, and magnitude std.
To encode samples use:
```
attention_vectors = model.encode(samples_collection,
                                 oids_list=['1', '2'],
                                 batch_size=1,
                                 concatenate=True)
```
where
- `samples_collection` is a list of numpy array light curves
- `oids_list` is a list with the light curves ids (needed to concatenate 200-len windows)
- `batch_size` specify the number of samples per forward pass
-  when `concatenate=True` ASTROMER concatenates every 200-lenght windows belonging the same object id. The output when `concatenate=True` is a list of vary-length attention vectors.

## Finetuning or training from scratch
`ASTROMER` can be easly trained by using the `fit`. It include

```
from ASTROMER import SingleBandEncoder

model = SingleBandEncoder(num_layers= 2,
                          d_model   = 256,
                          num_heads = 4,
                          dff       = 128,
                          base      = 1000,
                          dropout   = 0.1,
                          maxlen    = 200)
model.from_pretrained('macho')
```
where,
- `num_layers`: Number of self-attention blocks
- `d_model`: Self-attention block dimension (must be divisible by `num_heads`)
- `num_heads`: Number of heads within the self-attention block
- `dff`: Number of neurons for the fully-connected layer applied after the attention blocks
- `base`: Positional encoder base (see formula)
- `dropout`: Dropout applied to output of the fully-connected layer
- `maxlen`: Maximum length to process in the encoder
Notice you can ignore `model.from_pretrained('macho')` for clean training.
```
mode.fit(train_data,
         validation_data,
         epochs=2,
         patience=20,
         lr=1e-3,
         project_path='./my_folder',
         verbose=0)
```
where,
- `train_data`: Training data already formatted as tf.data
- `validation_data`: Validation data already formatted as tf.data
- `epochs`: Number of epochs for training
- `patience`: Early stopping patience
- `lr`: Learning rate
- `project_path`: Path for saving weights and training logs
- `verbose`: (0) Display information during training (1) don't

`train_data` and `validation_data` should be loaded using `load_numpy` or `pretraining_records` functions. Both functions are in the `ASTROMER.preprocessing` module.

For large datasets is recommended to use Tensorflow Records ([see this tutorial to execute our data pipeline](https://github.com/astromer-science/main-code/blob/main/presentation/notebooks/create_records.ipynb))

## Resources
- [ASTROMER Tutorials](https://www.stellardnn.org/astromer/)

## Contributing to ASTROMER 🤝
If you train your model from scratch, you can share your pre-trained weights by submitting a Pull Request on [the weights repository](https://github.com/astromer-science/weights)
