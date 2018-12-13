from keras import backend as K
from keras import activations, constraints, initializers, regularizers
from keras.engine import InputSpec, Layer
from keras.layers import Dense
from keras.utils import conv_utils


class FactorizedDense(Layer):
    """Just your regular densely-connected NN layer.

    This layer based on `keras.layers.core.Dense` and behave like it.
    `FactorizedDense` implements the operation:
    `output = activation(dot(dot(input, pre_kernel), post_kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `pre_kernel` and `post_kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `pre_kernel`.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        components: Positive integer or None, the size of internal components.
            If given None, the output is calculated as the same manner as `Dense` layer.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        pre_kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        post_kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        pre_kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        post_kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """
    target_layer_types = [Dense]

    def __init__(self, units, components,
                 activation=None,
                 use_bias=True,

                 pre_kernel_initializer='glorot_uniform',
                 post_kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',

                 pre_kernel_regularizer=None,
                 post_kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,

                 pre_kernel_constraint=None,
                 post_kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FactorizedDense, self).__init__(**kwargs)
        self.units = units
        self.components = components
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.pre_kernel_initializer = initializers.get(pre_kernel_initializer)
        self.post_kernel_initializer = initializers.get(post_kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.pre_kernel_regularizer = regularizers.get(pre_kernel_regularizer)
        self.post_kernel_regularizer = regularizers.get(post_kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.pre_kernel_constraint = constraints.get(pre_kernel_constraint)
        self.post_kernel_constraint = constraints.get(post_kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        is_factorized = self.components is not None

        if is_factorized:
            shape = (input_dim, self.components)
        else:
            shape = (input_dim, self.units)

        self.pre_kernel = self.add_weight(shape,
                                          initializer=self.pre_kernel_initializer,
                                          name='pre_kernel',
                                          regularizer=self.pre_kernel_regularizer,
                                          constraint=self.pre_kernel_constraint)

        if not is_factorized:
            self.post_kernel = None
        else:
            self.post_kernel = self.add_weight((self.components, self.units),
                                               initializer=self.post_kernel_initializer,
                                               name='kernel',
                                               regularizer=self.post_kernel_regularizer,
                                               constraint=self.post_kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight((self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        h = K.dot(inputs, self.pre_kernel)
        if self.post_kernel is not None:
            h = K.dot(h, self.post_kernel)
        if self.use_bias:
            h = K.bias_add(h, self.bias)
        if self.activation is not None:
            h = self.activation(h)
        return h

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'components': self.components,
            'use_bias': self.use_bias,

            'pre_kernel_initializer': initializers.serialize(self.pre_kernel_initializer),
            'post_kernel_initializer': initializers.serialize(self.post_kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),

            'pre_kernel_regularizer': regularizers.serialize(self.pre_kernel_regularizer),
            'post_kernel_regularizer': regularizers.serialize(self.post_kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),

            'pre_kernel_constraint': constraints.serialize(self.pre_kernel_constraint),
            'post_kernel_constraint': constraints.serialize(self.post_kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(FactorizedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FactorizedConv2DTucker(Layer):
    """2D convolution layer with tucker decomposition.


    This layer is based on `keras.layers.convolution.Conv2D` and behave like it.
    The difference is the kernel is factorized by tucker decomposition.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        input_components: Integer or None, the number of components
            of kernel for the input channel axis. If given None, the
            factorization of input side is skipped.
        output_components: Integer or None, the number of components
            of kernel for the output channel axis. If given None, the
            factorization of output side is skipped.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, width, height, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, width, height)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.

        pre_kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        post_kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).

        pre_kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        post_kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).

        pre_kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        post_kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 input_components=None,
                 output_components=None,

                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,

                 pre_kernel_initializer='glorot_uniform',
                 kernel_initializer='glorot_uniform',
                 post_kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',

                 pre_kernel_regularizer=None,
                 kernel_regularizer=None,
                 post_kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,

                 pre_kernel_constraint=None,
                 kernel_constraint=None,
                 post_kernel_constraint=None,
                 bias_constraint=None,

                 **kwargs):
        super(FactorizedConv2DTucker, self).__init__(**kwargs)
        rank = 2
        self.rank = rank
        self.input_components = input_components
        self.output_components = output_components
        self.filters = filters
        self.output_components = output_components

        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.pre_kernel_initializer = initializers.get(pre_kernel_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.post_kernel_initializer = initializers.get(post_kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.pre_kernel_regularizer = regularizers.get(pre_kernel_regularizer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.post_kernel_regularizer = regularizers.get(post_kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.pre_kernel_constraint = constraints.get(pre_kernel_constraint)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.post_kernel_constraint = constraints.get(post_kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(ndim=rank + 2)  # batch, H, W, C

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        if self.input_components is None:
            input_components = input_dim
        else:
            input_components = self.input_components
        if self.output_components is None:
            output_components = self.filters
        else:
            output_components = self.output_components
        kernel_shape = self.kernel_size + (input_components, output_components)

        if self.input_components is None:
            self.pre_kernel = None
        else:
            pre_kernel_shape = (1, 1) + (input_dim, self.input_components)
            self.pre_kernel = self.add_weight(pre_kernel_shape,
                                              initializer=self.pre_kernel_initializer,
                                              name='pre_kernel',
                                              regularizer=self.pre_kernel_regularizer,
                                              constraint=self.pre_kernel_constraint)

        self.kernel = self.add_weight(kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.output_components is None:
            self.post_kernel = None
        else:
            post_kernel_shape = (1, 1) + (self.output_components, self.filters)
            self.post_kernel = self.add_weight(post_kernel_shape,
                                               initializer=self.post_kernel_initializer,
                                               name='post_kernel',
                                               regularizer=self.post_kernel_regularizer,
                                               constraint=self.post_kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight((self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        h = inputs
        if self.pre_kernel is not None:
            h = K.conv2d(
                h,
                self.pre_kernel,
                strides=(1, 1),
                padding='valid',
                data_format=self.data_format,
                dilation_rate=(1, 1),
            )
        h = K.conv2d(
            h,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if self.post_kernel is not None:
            h = K.conv2d(
                h,
                self.post_kernel,
                strides=(1, 1),
                padding='valid',
                data_format=self.data_format,
                dilation_rate=(1, 1),
            )

        outputs = h
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'input_components': self.input_components,
            'output_components': self.output_components,
            'filters': self.filters,

            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,

            'pre_kernel_initializer': initializers.serialize(self.pre_kernel_initializer),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'post_kernel_initializer': initializers.serialize(self.post_kernel_initializer),
            'bias_initializer': initializers.serialize(self.kernel_initializer),

            'pre_kernel_regularizer': regularizers.serialize(self.pre_kernel_regularizer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'post_kernel_regularizer': regularizers.serialize(self.post_kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),

            'pre_kernel_constraint': constraints.serialize(self.pre_kernel_constraint),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'post_kernel_constraint': constraints.serialize(self.post_kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(FactorizedConv2DTucker, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_layers = {
    'FactorizedConv2DTucker': FactorizedConv2DTucker,
    'FactorizedDense': FactorizedDense,
}
