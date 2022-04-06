# ASTROMER Python library 🔭
### Release: v0.0.1 (the origin 👁️)

ASTROMER is a transformer based model pretrained on millions of light curves. ASTROMER can be finetuned on specific datasets to create useful representations that can improve the performance of novel deep learning models.

❗ This version of ASTROMER can only works on single band light curves. 

🔥 [See the official repo here](https://github.com/astromer-science/main-code) 

## Install
```
pip install deepweights
```

## How to use it
Currently, there are 2 pre-trained models: `macho` and `atlas`.
To load weights use:
```
from deepweights import ASTROMER

model = ASTROMER()
model.from_pretrained('macho')
```
It will automatically download the weights from [this public github repository](https://github.com/astromer-science/weights.git) and load them into the `ASTROMER()` instance.

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
-  when `concatenate=True` ASTROMER concatenates every 200-lenght windows belonging the same object id (remember this version of ASTROMER only works up to 200 observations). The output when `concatenate=True` is a list of vary-length attention vectors.

## Finetuning or training from scratch
`ASTROMER` is a [Tensorflow custom model](https://www.tensorflow.org/guide/keras/custom_layers_and_models#the_model_class). It means we can use the [`fit` method](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) and [`callbacks`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks) as usual.

```
from deepweights import ASTROMER

model = ASTROMER(num_layers= 2,
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
model.compile(optimizer='adam')
mode.fit(data, epochs=100, callbacks=[], ...)
mode.save()
```
We recomend to use the `CustomSchedule` class to control the learning rate during training.
i.e., 
```
from deepweights import CustomSchedule
learning_rate = CustomSchedule(d_model)
optimizer     = tf.keras.optimizers.Adam(learning_rate, 
                                         beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer)
```
For large datasets is recommended to use Tensorflow Records ([see this tutorial to execute our data pipeline](https://github.com/astromer-science/main-code/blob/main/presentation/notebooks/create_records.ipynb)) 

## Contributing to ASTROMER 🤝
If you train your model from scratch, you can share your pre-trained weights by submitting a Pull Request on [the weights repository](https://github.com/astromer-science/weights)

