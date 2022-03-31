import tensorflow as tf

from tensorflow.keras.layers import Input, Layer, Dense, LayerNormalization


class RegLayer(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.reg_layer = Dense(1, name='RegLayer')
		self.bn_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs, training=False):
		x = self.bn_0(inputs)
		x = self.reg_layer(x, training=training)
		return x

class NSP_Regressor(Layer):
	def __init__(self, **kwargs):
		super(NSP_Regressor, self).__init__(**kwargs)
		self.reg_layer = Dense(1, name='RegLayer')
		self.clf_layer = Dense(2, name='CLFLayer')
		self.layernorm = LayerNormalization(epsilon=1e-6)

	def call(self, inputs, training=False):
		nsp_tokens = tf.slice(inputs, [0,0,0],[-1, 1,-1])
		nsp_shp = tf.shape(nsp_tokens)
		nsp_tokens = tf.reshape(nsp_tokens, [nsp_shp[0], nsp_shp[-1]])
		rec_tokens = tf.slice(inputs, [0,1,0],[-1, -1, -1])

		rec_tokens = self.layernorm(rec_tokens)
		reconstruction = self.reg_layer(rec_tokens)
		nsp_predicted  = self.clf_layer(nsp_tokens)
		return reconstruction, nsp_predicted
