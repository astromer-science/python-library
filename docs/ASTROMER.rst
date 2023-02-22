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



.. automodule:: ASTROMER.models
   :members:
   :undoc-members:
   :show-inheritance:


Preprocessing
============================

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

Module contents
---------------

.. automodule:: ASTROMER
   :members:
   :undoc-members:
   :show-inheritance:
