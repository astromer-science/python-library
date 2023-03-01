ASTROMER
****************************

Models
=============================

Single-Band Encoder
=============================

.. epigraph:: The **Single-Band Encoder** represents the main class of the models, which load, fit, encode and train the preprocessed weights.

.. epigraph:: It took every single-band light curve that may vary between different stars, and this depends on the objectives of the survey being carried out. 
.. epigraph:: The X is a set of observations of a celestial object over time (such as a star). Each observation had two characteristics: the magnitude (brightness) of the object and the Modified Julian Date (MJD) when the observation was made.

.. epigraph:: We propose to use learned representations of a transformer-based encoder to create embeddings that represent the variability of objects in dk.dimensional space. Making easy to fine-tune the model weights to match other surveys and use them to solve downstream task, such as classification or regression.

.. automodule:: ASTROMER.models
   :members:
   :show-inheritance: 


Preprocessing
============================

.. epigraph::This method creates the ASTRONOMER input format, which is based on the BERT masking strategy (Devlin et al., 2018) and results in all samples having the same length.

.. automodule:: ASTROMER.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Utils
============================

.. automodule:: ASTROMER.utils
   :members:
   :undoc-members:
   :show-inheritance:

Quick-start
============================

.. epigraph:: **Install**

.. epigraph:: First, install the ASTROMER wheel using pip

.. code-block:: python
   
   pip install ASTROMER
   
.. code-block:: python

   from ASTROMER.models import SingleBandEncoder

.. epigraph::
  Then initiate

.. code-block:: python

  model = SingleBandEncoder()
  model = model.from_pretraining('macho')

.. epigraph:: It will automatically download the weights from this public github repository and load them into the SingleBandEncoder instance. Assuming you have a list of vary-lenght (numpy) light curves.

.. code-block:: python

  import numpy as np

  samples_collection = [ np.array([[5200, 0.3, 0.2],
                                   [5300, 0.5, 0.1],
                                   [5400, 0.2, 0.3]]),


.. epigraph:: Light curves are Lx3 matrices with time, magnitude, and magnitude std. To encode samples use:

.. code-block:: python


  attention_vectors = model.encode(samples_collection,
                                    oids_list=['1', '2'],
                                    batch_size=1,
                                    concatenate=True)


Fine Tune
-----------------------------------------------

.. epigraph:: `ASTROMER` can be easly trained by using the `fit`. It include

.. code-block:: python
   
   from ASTROMER import SingleBandEncoder

   model = SingleBandEncoder(num_layers= 2,
                          d_model   = 256,
                          num_heads = 4,
                          dff       = 128,
                          base      = 1000,
                          dropout   = 0.1,
                          maxlen    = 200)
                   
   model.from_pretrained('macho')

.. epigraph:: where,

- `num_layers`: Number of self-attention blocks
- `d_model`: Self-attention block dimension (must be divisible by `num_heads`)
- `num_heads`: Number of heads within the self-attention block
- `dff`: Number of neurons for the fully-connected layer applied after the attention blocks
- `base`: Positional encoder base (see formula)
- `dropout`: Dropout applied to output of the fully-connected layer
- `maxlen`: Maximum length to process in the encoder

.. epigraph:: Notice you can ignore `model.from_pretrained('macho')` for clean training.

.. code-block:: python

   mode.fit(train_data,
         validation_data,
         epochs=2,
         patience=20,
         lr=1e-3,
         project_path='./my_folder',
         verbose=0)

.. epigraph:: where,

- `train_data`: Training data already formatted as tf.data
- `validation_data`: Validation data already formatted as tf.data
- `epochs`: Number of epochs for training
- `patience`: Early stopping patience
- `lr`: Learning rate
- `project_path`: Path for saving weights and training logs
- `verbose`: (0) Display information during training (1) don't

`train_data` and `validation_data` should be loaded using `load_numpy` or `pretraining_records` functions. Both functions are in the `ASTROMER.preprocessing` module.

.. epigraph:: For large datasets is recommended to use Tensorflow Records `see this tutorial to execute our data pipeline <https://github.com/astromer-science/main-code/blob/main/presentation/notebooks/create_records.ipynb>`_

