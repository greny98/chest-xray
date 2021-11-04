from tensorflow.keras import layers, Model
from backbone.model import load_basenet
from static_values.values import IMAGE_SIZE


def get_backbone(weights=None):
    base_net = load_basenet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    if weights is not None:
        base_net.load_weights(weights).expect_partial()
        for l in base_net.layers:
            l.trainable = False

    extract_layers = ['pool2_pool', 'pool3_pool', 'pool4_pool']
    feature_maps = [base_net.get_layer(name).output for name in extract_layers]
    return Model(inputs=[base_net.inputs], outputs=feature_maps)


def pyramid_block(l_layers):
    out_layers = []
    for i in range(len(l_layers) - 2, -1, -1):
        upscale = layers.UpSampling2D(2)(l_layers[i + 1])
        out = layers.Add()([l_layers[i], upscale])
        out_layers.append(out)
    out_layers = out_layers[::-1]
    out_layers.append(l_layers[-1])
    return out_layers


def down_sampling(pyr_out, name):
    pyr_out = layers.Conv2D(256, 3, 1, padding='same', name=name + '_conv')(pyr_out)
    pyr_out = layers.BatchNormalization(epsilon=1.001e-5, name=name + '_bn')(pyr_out)
    pyr_out = layers.Activation('relu', name=name + '_relu')(pyr_out)
    pyr_out = layers.MaxPooling2D(name=name + '_pool')(pyr_out)
    return pyr_out


def FeaturePyramid(backbone: Model):
    pool_out1, pool_out2, pool_out3 = backbone.outputs
    # Change all to 256 units
    pyr_out1 = layers.Conv2D(256, 1, name='pyr_out1_conv1')(pool_out1)
    pyr_out2 = layers.Conv2D(256, 1, name='pyr_out2_conv1')(pool_out2)
    pyr_out3 = layers.Conv2D(256, 1, name='pyr_out3_conv1')(pool_out3)
    pyr_out4 = down_sampling(pyr_out3, name='pyr_out4')
    pyr_out5 = down_sampling(pyr_out4, name='pyr_out5')
    # pyramid handle
    pyr_out1, pyr_out2, pyr_out3, pyr_out4, pyr_out5 = pyramid_block(
        [pyr_out1, pyr_out2, pyr_out3, pyr_out4, pyr_out5])
    # after pyramid
    pyr_out1 = layers.Conv2D(256, 3, 1, padding='same', name='pyr_out1_conv2')(pyr_out1)
    pyr_out2 = layers.Conv2D(256, 3, 1, padding='same', name='pyr_out2_conv2')(pyr_out2)
    pyr_out3 = layers.Conv2D(256, 3, 1, padding='same', name='pyr_out3_conv2')(pyr_out3)
    pyr_out4 = layers.Conv2D(256, 3, 1, padding='same', name='pyr_out4_conv2')(pyr_out4)
    pyr_out5 = layers.Conv2D(256, 3, 1, padding='same', name='pyr_out5_conv2')(pyr_out5)
    return Model(inputs=[backbone.inputs],
                 outputs=[pyr_out1, pyr_out2, pyr_out3, pyr_out4, pyr_out5])
