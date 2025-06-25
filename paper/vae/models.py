import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Lambda,
    Input,
    Dense,
    Conv2D,
    AveragePooling2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Reshape,
    Activation,
    LeakyReLU
    )
#from qkeras import (
#    QConv2D,
#    QDense,
#    QActivation
#    )
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
#from qkeras import QConv2D, QDense, QActivation

from losses import (
    make_mse_kl,
    make_mse,
    make_kl
    )

# number of integer bits for each bit width
QUANT_INT = {
    0: 0,
    2: 1,
    4: 2,
    6: 2,
    8: 3,
    10: 3,
    12: 4,
    14: 4,
    16: 6
    }

def model_set_weights(model, load_model, quant_size):
   # load trained model
    with open(load_model+'.json', 'r') as jsonfile:
        config = jsonfile.read()
    bp_model = tf.keras.models.model_from_json(config,
        custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
        'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation})
    bp_model.load_weights(load_model+'.h5')

    # set weights for encoder and skip input quantization
    if quant_size!=0:
        for i, _ in enumerate(model.layers[1].layers):
                if i < 2: continue
                model.layers[1].layers[i].set_weights(bp_model.layers[1].layers[i-1].get_weights())
    else:
        for i, _ in enumerate(model.layers[1].layers):
                model.layers[1].layers[i].set_weights(bp_model.layers[1].layers[i].get_weights())
    # set weights for decoder
    for i, _ in enumerate(model.layers[2].layers):
        model.layers[2].layers[i].set_weights(bp_model.layers[2].layers[i].get_weights())
    return model



def mse_loss_tf(inputs, outputs):
    return tf.math.reduce_mean(tf.math.square(outputs - inputs), axis=-1)

def make_mse_loss(inputs, outputs):
    # remove last dimension
    inputs = tf.reshape(inputs, (tf.shape(inputs)[0],19,3,1))
    outputs = tf.reshape(outputs, (tf.shape(outputs)[0],19,3,1))

    inputs = tf.squeeze(inputs, axis=-1)
    inputs = tf.cast(inputs, dtype=tf.float32)
    # trick with phi
    outputs_phi = math.pi*tf.math.tanh(outputs)
    # trick with phi
    outputs_eta_egamma = 3.0*tf.math.tanh(outputs)
    outputs_eta_muons = 2.1*tf.math.tanh(outputs)
    outputs_eta_jets = 4.0*tf.math.tanh(outputs)
    outputs_eta = tf.concat([outputs[:,0:1,:], outputs_eta_egamma[:,1:5,:], outputs_eta_muons[:,5:9,:], outputs_eta_jets[:,9:19,:]], axis=1)
    # use both tricks
    outputs = tf.concat([outputs[:,:,0], outputs_eta[:,:,1], outputs_phi[:,:,2]], axis=2)
    # mask zero features
    mask = tf.math.not_equal(inputs,0)
    mask = tf.cast(mask, tf.float32)
    outputs = mask * outputs
    loss = mse_loss_tf(tf.reshape(inputs, [-1, 57]), tf.reshape(outputs, [-1, 57]))
    loss = tf.math.reduce_mean(loss, axis=0) # average over batch
    return loss

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.total_val_loss_tracker = metrics.Mean(name="total_val_loss")
        self.reconstruction_val_loss_tracker = metrics.Mean(name="reconstruction_val_loss")
        self.kl_val_loss_tracker = metrics.Mean(name="kl_val_loss")

    def call(self, inputs):
        """Defines the forward pass of the VAE."""
        mu, logvar, _ = self.encoder(inputs)
        return tf.concat([mu, logvar], axis=-1)  # Concatenate mu and logvar

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.total_val_loss_tracker,
            self.reconstruction_val_loss_tracker,
            self.kl_val_loss_tracker
        ]

    def train_step(self, data):
        data_in, target = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data_in, training=True)
            reconstruction = self.decoder(z, training=True)

            reconstruction_loss = make_mse_loss(target, reconstruction)
            beta = 0.8
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(kl_loss, axis=-1)
            total_loss = (1 - beta) * reconstruction_loss + beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state((1 - beta) * reconstruction_loss)
        self.kl_loss_tracker.update_state(beta * kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        data_in, target = data
        z_mean, z_log_var, z = self.encoder(data_in)
        reconstruction = self.decoder(z)

        reconstruction_loss = make_mse_loss(target, reconstruction)
        beta = 0.8
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(kl_loss, axis=-1)
        total_loss = (1 - beta) * reconstruction_loss + beta * kl_loss

        self.total_val_loss_tracker.update_state(total_loss)
        self.reconstruction_val_loss_tracker.update_state((1 - beta) * reconstruction_loss)
        self.kl_val_loss_tracker.update_state(beta * kl_loss)

        return {
            "loss": self.total_val_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_val_loss_tracker.result(),
            "kl_loss": self.kl_val_loss_tracker.result(),
        }

    def save(self, path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        print('Saving model to {}'.format(path))
        self.encoder.save(os.path.join(path, 'encoder.h5'))
        self.decoder.save(os.path.join(path, 'decoder.h5'))

    @classmethod
    def load(cls, path, custom_objects={}):
        encoder = tf.keras.models.load_model(
            os.path.join(path, 'encoder.h5'), custom_objects=custom_objects, compile=False
        )
        decoder = tf.keras.models.load_model(
            os.path.join(path, 'decoder.h5'), custom_objects=custom_objects, compile=False
        )
        return cls(encoder, decoder)

    def get_config(self):
        return {
            "encoder": tf.keras.layers.serialize(self.encoder),
            "decoder": tf.keras.layers.serialize(self.decoder),
        }

    @classmethod
    def from_config(cls, config):
        encoder = tf.keras.layers.deserialize(config["encoder"])
        decoder = tf.keras.layers.deserialize(config["decoder"])
        return cls(encoder, decoder)

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def dense_vae():

    # Define autoencoder model
    latent_dim = 3
    input_shape = 57

    #encoder
    inputArray = Input(shape=(57,))
    x = BatchNormalization()(inputArray)
    x = Dense(32, kernel_initializer=tf.keras.initializers.HeUniform(seed=42))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(16, kernel_initializer=tf.keras.initializers.HeUniform(seed=42))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    mu = Dense(latent_dim, name = 'latent_mu', kernel_initializer=tf.keras.initializers.HeUniform(seed=42))(x)
    logvar = Dense(latent_dim, name = 'latent_logvar', kernel_initializer=tf.keras.initializers.HeUniform(seed=42))(x)

    # Use reparameterization trick to ensure correct gradient
    z = Sampling()([mu, logvar])

    # Create encoder
    encoder = Model(inputArray, [mu, logvar, z], name='encoder')
    encoder.summary()

    #decoder
    d_input = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(16, kernel_initializer=tf.keras.initializers.HeUniform(seed=42))(d_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(32, kernel_initializer=tf.keras.initializers.HeUniform(seed=42))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    dec = Dense(input_shape, kernel_initializer=tf.keras.initializers.HeUniform(seed=42))(x)

    # Create decoder
    decoder = Model(d_input, dec, name='decoder')
    decoder.summary()

    # Compile model
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=Adam())
    return vae

def sample_z(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    eps = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(z_log_var / 2) * eps

class conv_vae(Model):
    def __init__(self, image_shape, latent_dim, beta=1.0, quant_size=0, **kwargs):
        super(conv_vae, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.beta = beta
        self.image_shape = image_shape
        self.quant_size = quant_size

        # Encoder
        input_encoder = Input(shape=image_shape[1:], name='encoder_input')
        x = ZeroPadding2D(((1, 0), (0, 0)))(input_encoder)
        x = BatchNormalization()(x)
        x = Conv2D(16, kernel_size=(3, 3), use_bias=False, padding='valid')(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(3, 1))(x)
        x = Conv2D(32, kernel_size=(3, 1), use_bias=False, padding='same')(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(3, 1))(x)
        x = Flatten()(x)
        z_mean = Dense(latent_dim, name='latent_mu')(x)
        z_log_var = Dense(latent_dim, name='latent_sigma')(x)
        z = Lambda(self.sample_z, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        self.encoder = Model(input_encoder, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        input_decoder = Input(shape=(latent_dim,), name='decoder_input')
        x = Dense(64)(input_decoder)
        x = Activation('relu')(x)
        x = Reshape((2, 1, 32))(x)
        x = Conv2D(32, kernel_size=(3, 1), use_bias=False, padding='same')(x)
        x = Activation('relu')(x)
        x = UpSampling2D((3, 1))(x)
        x = ZeroPadding2D(((0, 0), (1, 1)))(x)
        x = Conv2D(16, kernel_size=(3, 1), use_bias=False, padding='same')(x)
        x = Activation('relu')(x)
        x = UpSampling2D((3, 1))(x)
        x = ZeroPadding2D(((1, 0), (0, 0)))(x)
        dec = Conv2D(1, kernel_size=(3, 3), use_bias=False, padding='same')(x)

        self.decoder = Model(input_decoder, dec, name='decoder')

        # Loss trackers
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")


    def sample_z(self, args):
        """Sampling function for latent variable."""
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean), mean=0.0, stddev=1.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs, training=False):
        """Forward pass through the VAE."""
        # Encoder forward pass
        encoder_outputs = self.encoder(inputs, training=training)
        z_mean, z_log_var, z = encoder_outputs

        # Decoder forward pass
        reconstruction = self.decoder(z, training=training)

        # Return only reconstruction during inference
        if not training:
            return reconstruction

        # Return all outputs during training
        return reconstruction, z_mean, z_log_var

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        # Unpack inputs and targets
        x, y = data

        with tf.GradientTape() as tape:
            reconstruction, z_mean, z_log_var = self(x, training=True)

            # Compute losses
            reconstruction_loss = tf.reduce_mean(tf.square(reconstruction - y))
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = (1 - self.beta) * reconstruction_loss + self.beta * kl_loss

        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update loss trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        # Unpack inputs and targets
        x, y = data

        # Forward pass
        reconstruction = self(x, training=False)

        # Compute reconstruction loss
        reconstruction_loss = tf.reduce_mean(tf.square(reconstruction - y))

        # Update loss trackers
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.total_loss_tracker.update_state(reconstruction_loss)  # No KL loss in test_step

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }


    def get_config(self):
        config = super(conv_vae, self).get_config()
        config.update({
            "image_shape": self.image_shape,
            "latent_dim": self.latent_dim,
            "beta": self.beta,
            "quant_size": self.quant_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Remove explicitly defined arguments from `**kwargs`
        config = config.copy()
        config.pop("name", None)  # Remove attributes added by Keras if not needed
        config.pop("trainable", None)  # Remove default attributes added by Keras
        return cls(
            image_shape=config.pop("image_shape"),
            latent_dim=config.pop("latent_dim"),
            beta=config.pop("beta", 1.0),
            quant_size=config.pop("quant_size", 0),
            **config
        )
# def conv_vae(image_shape, latent_dim, beta, quant_size=0, pruning='not_pruned'):

#     int_size = QUANT_INT[quant_size]
#     # encoder
#     input_encoder = Input(shape=image_shape[1:], name='encoder_input')

#     if quant_size!=0:
#         quantized_inputs = QActivation('quantized_bits(16,10,0,alpha=1)')(input_encoder)
#         x = ZeroPadding2D(((1,0),(0,0)))(quantized_inputs)
#     else:
#         quantized_inputs = None
#         x = ZeroPadding2D(((1,0),(0,0)))(input_encoder)
#     #
#     x = BatchNormalization()(x)
#     #
#     x = Conv2D(16, kernel_size=(3,3), use_bias=False, padding='valid')(x) if quant_size==0 \
#         else QConv2D(16, kernel_size=(3,3), use_bias=False, padding='valid',
#                          kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
#     #
#     x = Activation('relu')(x) if quant_size==0 \
#         else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)
#     #
#     x = AveragePooling2D(pool_size=(3, 1))(x)
#     #
#     x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x) if quant_size==0 \
#         else QConv2D(32, kernel_size=(3,1), use_bias=False, padding='same',
#                          kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
#     #
#     x = Activation('relu')(x) if quant_size==0 \
#         else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)
#     #
#     x = AveragePooling2D(pool_size=(3, 1))(x)
#     #
#     x = Flatten()(x)
#     #
#     z_mean = Dense(latent_dim, name='latent_mu')(x) if quant_size==0 \
#         else QDense(latent_dim, name='latent_mu',
#                kernel_quantizer='quantized_bits(16,6,0,alpha=1)',
#                bias_quantizer='quantized_bits(16,6,0,alpha=1)')(x)

#     z_log_var = Dense(latent_dim, name='latent_sigma')(x) if quant_size==0 \
#         else QDense(latent_dim, name='latent_sigma',
#                kernel_quantizer='quantized_bits(16,6,0,alpha=1)',
#                bias_quantizer='quantized_bits(16,6,0,alpha=1)')(x)

#     z = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([z_mean, z_log_var])

#     encoder = Model(inputs=input_encoder, outputs=[z_mean, z_log_var, z], name='encoder_CNN')
#     if pruning=='pruned':
#         ''' How to estimate the enc step:
#             num_images = input_train.shape[0] * (1 - validation_split)
#             end_step = np.ceil(num_images / batch_size).astype(np.int32) * pruning_epochs
#             start at 5: np.ceil(14508274/2*0.8/1024).astype(np.int32) * 5 = 28340
#             stop at 15: np.ceil(14508274/2*0.8/1024).astype(np.int32) * 15 = 85020
#         '''
#         start_pruning = np.ceil(image_shape[0]*0.8/1024).astype(np.int32) * 5
#         end_pruning = np.ceil(image_shape[0]*0.8/1024).astype(np.int32) * 15
#         pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
#                                 initial_sparsity=0.0, final_sparsity=0.5,
#                                 begin_step=start_pruning, end_step=end_pruning)
#         encoder_pruned = tfmot.sparsity.keras.prune_low_magnitude(encoder, pruning_schedule=pruning_schedule)
#         encoder = encoder_pruned
#     encoder.summary()

#     # decoder
#     input_decoder = Input(shape=(latent_dim,), name='decoder_input')

#     x = Dense(64)(input_decoder)
#     #
#     x = Activation('relu')(x)
#     #
#     x = Reshape((2,1,32))(x)
#     #
#     x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x)
#     #
#     x = Activation('relu')(x)
#     #
#     x = UpSampling2D((3,1))(x)
#     x = ZeroPadding2D(((0,0),(1,1)))(x)
#     #
#     x = Conv2D(16, kernel_size=(3,1), use_bias=False, padding='same')(x)
#     x = Activation('relu')(x)
#     #
#     x = UpSampling2D((3,1))(x)
#     x = ZeroPadding2D(((1,0),(0,0)))(x)
#     #
#     dec = Conv2D(1, kernel_size=(3,3), use_bias=False, padding='same')(x)
#     #
#     decoder = Model(inputs=input_decoder, outputs=dec)
#     decoder.summary()
#     # vae
#     vae_outputs = decoder(encoder(input_encoder)[2])
#     vae = Model(input_encoder, vae_outputs, name='vae')
#     vae.summary()
#     # load weights
#     if pruning=='pruned':
#         vae = model_set_weights(vae, f'output/model-conv_vae-8-b0.8-q0-not_pruned', quant_size)
#     # compile VAE
#     vae.compile(optimizer=Adam(learning_rate=3E-3, amsgrad=True),
#                 loss=make_mse_kl(z_mean, z_log_var, beta),
#                 # metrics=[make_mse, make_kl(z_mean, z_log_var)]
#                 )
#     return vae

def conv_ae(image_shape, latent_dim, quant_size=0, pruning='not_pruned'):
    int_size = QUANT_INT[quant_size]
    # encoder
    input_encoder = Input(shape=image_shape[1:], name='encoder_input')
    if quant_size!=0:
        quantized_inputs = QActivation(f'quantized_bits(16,10,0,alpha=1)')(input_encoder)
        x = ZeroPadding2D(((1,0),(0,0)))(quantized_inputs)
    else:
        quantized_inputs = None
        x = ZeroPadding2D(((1,0),(0,0)))(input_encoder)
    x = BatchNormalization()(x)
    #
    x = Conv2D(16, kernel_size=(3,3), use_bias=False, padding='valid')(x) if quant_size==0 \
        else QConv2D(16, kernel_size=(3,3), use_bias=False, padding='valid',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = AveragePooling2D(pool_size=(3, 1))(x)
    #
    x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(32, kernel_size=(3,1), use_bias=False, padding='same',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = AveragePooling2D(pool_size=(3, 1))(x)
    #
    x = Flatten()(x)
    #
    enc = Dense(latent_dim)(x) if quant_size==0 \
        else QDense(latent_dim,
               kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)',
               bias_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)

    encoder = Model(inputs=input_encoder, outputs=enc)
    encoder.summary()
    # decoder
    input_decoder = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(64)(input_decoder) if quant_size==0 \
        else QDense(64,
               kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)',
               bias_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(input_decoder)
    #
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)
    #
    x = Reshape((2,1,32))(x)
    #
    x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(32, kernel_size=(3,1), use_bias=False, padding='same',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = UpSampling2D((3,1))(x)
    x = ZeroPadding2D(((0,0),(1,1)))(x)

    x = Conv2D(16, kernel_size=(3,1), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(16, kernel_size=(3,1), use_bias=False, padding='same',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = UpSampling2D((3,1))(x)
    x = ZeroPadding2D(((1,0),(0,0)))(x)

    dec = Conv2D(1, kernel_size=(3,3), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(1, kernel_size=(3,3), use_bias=False, padding='same',
                        kernel_quantizer='quantized_bits(16,10,0,alpha=1)')(x)
    #
    decoder = Model(inputs=input_decoder, outputs=dec)
    decoder.summary()

    if pruning=='pruned':
        start_pruning = np.ceil(image_shape[0]*0.8/1024).astype(np.int32) * 5
        end_pruning = np.ceil(image_shape[0]*0.8/1024).astype(np.int32) * 15
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                                initial_sparsity=0.0, final_sparsity=0.5,
                                begin_step=start_pruning, end_step=end_pruning)
        encoder_pruned = tfmot.sparsity.keras.prune_low_magnitude(encoder, pruning_schedule=pruning_schedule)
        encoder = encoder_pruned
        decoder_pruned = tfmot.sparsity.keras.prune_low_magnitude(decoder, pruning_schedule=pruning_schedule)
        decoder = decoder_pruned

    # ae
    ae_outputs = decoder(encoder(input_encoder))
    autoencoder = Model(inputs=input_encoder, outputs=ae_outputs)
    autoencoder.summary()
    # load weights
    if pruning=='pruned':
        autoencoder = model_set_weights(autoencoder, f'output/model-conv_ae-8-b0-q0-not_pruned', quant_size)
    # compile AE
    autoencoder.compile(optimizer=Adam(lr=3E-3, amsgrad=True),
        loss=make_mse)
    return autoencoder
