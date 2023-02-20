ASTROMER
****************************

Models
============================


Single-Band Encoder
-----------------
.. epigraph::
The **Single-Band Encoder** represents the main class of the models, which load, fit, encode and train the preprocessed weights.

.. epigraph::
The Single-Band encoder took every single-band light curve that may vary between different stars, and this depends on the objectives of the survey being carried out.
The X is a set of observations of a celestial object over time (such as a star). Each observation had two characteristics: the magnitude (brightness) of the object and the Modified Julian Date (MJD) when the observation was made.

.. epigraph::
We propose to use learned representations of a transformer-based encoder to create embeddings that represent the variability of objects in dk.dimensional space.
We can fine-tune the model weights to match other surveys and use them to solve downstream task, such as classification or regression.

.. automodule:: src.models
   :members:
   :undoc-members:
   :show-inheritance:

Utils
============================

.. automodule:: src.utils
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing
============================

.. automodule:: src.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:



QUICK START
============================

**Install**

.. epigraph::
  First, install the ASTROMER wheel using pip

.. code-block:: python

  !pip install ASTROMER
  from ASTROMER.models import SingleBandEncoder

.. epigraph::
  Then initiate

.. code-block:: python

  model = SingleBandEncoder()
  model = model.from_pretraining('macho')
.. epigraph::
It will automatically download the weights from this public github repository and load them into the SingleBandEncoder instance.
Assuming you have a list of vary-lenght (numpy) light curves.


.. code-block:: python

  import numpy as np

  samples_collection = [ np.array([[5200, 0.3, 0.2],
                                   [5300, 0.5, 0.1],
                                   [5400, 0.2, 0.3]]),


.. epigraph::
Light curves are Lx3 matrices with time, magnitude, and magnitude std. To encode samples use:

.. code-block:: python


  attention_vectors = model.encode(samples_collection,
                                    oids_list=['1', '2'],
                                    batch_size=1,
                                    concatenate=True)


Fine Tune
----------------