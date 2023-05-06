import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


"""
Re-Implementation of the Conv_DCFD script

"""


def get_bases(kernel_size, num_bases):
    """
    get the fourier bessel bases. This uses the function from the fb.py file
    inputs: 
    kernel size
    number of bases

    outputs: (num_bases x kernel_size**2) tensor

    TODO (eventually): Dont hardcode the Fourier Bessel bases. Allow it to work with different shapes

    """
    if kernel_size == 3 and num_bases == 6:
        res = tf.convert_to_tensor([[ 4.9333e-02,  3.0530e-01,  4.9333e-02,  3.0530e-01,  6.7311e-01,
              3.0530e-01,  4.9333e-02,  3.0530e-01,  4.9333e-02],
            [-7.8269e-02, -5.9284e-01, -7.8269e-02,  3.6301e-17,  0.0000e+00,
              3.6301e-17,  7.8269e-02,  5.9284e-01,  7.8269e-02],
            [-7.8269e-02,  7.2602e-17,  7.8269e-02, -5.9284e-01,  0.0000e+00,
              5.9284e-01, -7.8269e-02,  0.0000e+00,  7.8269e-02],
            [-2.7126e-17,  6.7993e-01, -2.7126e-17, -6.7993e-01,  0.0000e+00,
             -6.7993e-01,  9.0420e-18,  6.7993e-01,  9.0420e-18],
            [ 1.4767e-01, -1.6654e-16, -1.4767e-01, -8.3268e-17,  0.0000e+00,
              8.3268e-17, -1.4767e-01,  0.0000e+00,  1.4767e-01],
            [-1.1172e-01, -4.0881e-01, -1.1172e-01, -4.0881e-01,  1.0270e+00,
             -4.0881e-01, -1.1172e-01, -4.0881e-01, -1.1172e-01]])
    else:
        raise ValueError("Only a kernel size of 3 and num bases of 6 is allowed")
    return res



class Conv_DCFD_tf(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, inter_kernel_size=5, stride=1, padding="SAME", num_bases=6, bias=True):
        super(Conv_DCFD_tf, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.inter_kernel_size = inter_kernel_size
        self.stride = stride
        self.padding = padding
        self.num_bases = num_bases
        

        bases = get_bases(kernel_size, num_bases)
        self.bases_len = len(bases)

        # set define the trainable variables
        self.bases = tf.Variable(bases, trainable=True, name='bases')
        self.coef = tf.Variable(tf.random.normal((1, 1, in_channels*num_bases, out_channels)), trainable=True)
        
        self.has_bias = bias

        # define these in the build part
        self.bias = None
        self.bases_model = None


    def get_config(self):
        config = super(Conv_DCFD_tf, self).get_config()
        # used to save and load the model
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'num_bases': self.num_bases,
            'bias': self.has_bias
        })
        return config
        
    def build(self, input_shape):
        """
        Apparently you need to build the layer in Tensorflow sometimes. 
        The same thing can be done in the __init__ function, but tensorflow is sometimes picky.
        """

        bases_size = self.num_bases * self.bases_len

        inter = max(64, bases_size//2)
        self.bases_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(inter, kernel_size=3, padding='same', strides=self.stride),
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(bases_size, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.Activation('relu')
        ])


        if self.has_bias:
            self.bias = self.add_weight(name='bias',
                                         shape=(self.out_channels),
                                         initializer='zeros',
                                         trainable=True)
        else:
            self.bias = None


    
    def call(self, input):
        N, H, W, C = input.shape # tf images are shaped (#, height, width, channel)
        H = int(H/self.stride)
        W = int(W/self.stride)

        M = self.num_bases

        drop_rate = 0.0
        
        # get the bases from what was shown in build
        bases = self.bases_model(input)

         
        bases = tf.reshape(bases, shape=(tf.shape(input)[0], H, W, self.num_bases, self.bases_len))

        # Do matrix multiplication on the bases from the network and the bases weights
        bases = tf.einsum('bhwtk, kl -> bhwtl', bases, self.bases)

        # Extrac the patches from the input
        x = tf.image.extract_patches(input, sizes=[1, self.kernel_size, self.kernel_size, 1], strides=[1, self.stride, self.stride, 1], rates=[1, 1, 1, 1], padding=self.padding)
        x = tf.reshape(x, [tf.shape(input)[0], H, W, self.in_channels, self.kernel_size*self.kernel_size])


        # do this again
        bases_reshaped = tf.reshape(bases, [tf.shape(input)[0], H, W, self.num_bases, -1])
        bases_out = tf.einsum('bhwml, bhwcl-> bhwcm', bases_reshaped, x)
        bases_out = tf.reshape(bases_out, [tf.shape(input)[0], H, W, self.in_channels * self.num_bases])

        # final convolutional layer, uses trainable coeficients
        out = tf.nn.conv2d(bases_out, self.coef, strides=[1,1,1,1], padding='VALID')
        out = tf.nn.bias_add(out, self.bias) # add the bias term

        return out




if __name__ == "__main__":
    # Define the model
    model = tf.keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        Conv_DCFD_tf(in_channels=3, out_channels=64, kernel_size=3, inter_kernel_size=5, stride=2, num_bases=6),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(12, (3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(units=256, activation='relu'),
        layers.Dense(units=10, activation='softmax')
    ])
    model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    model.summary()

