from tensorflow.keras import layers, Model
from backbone.model import load_basenet
from static_values.values import IMAGE_SIZE


def get_backbone(weights=None):
    base_net = load_basenet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    if weights is not None:
        base_net.load_weights(weights).expect_partial()
    last_out = layers.MaxPooling2D()(base_net.output)

    extract_layers = ['pool2_pool', 'pool3_pool', 'pool4_pool']
    feature_maps = [base_net.get_layer(name).output for name in extract_layers]
    feature_maps.append(last_out)
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


def FeaturePyramid(backbone: Model):
    backbone.load_weights('ckpt/checkpoint')
    # freeze backbone
    for l in backbone.layers:
        l.trainable = False
    # pool_out1=(40x40) - pool_out2=(20x20) - pool_out3=(10x10) - pool_out4=(5x5)
    pool_out1, pool_out2, pool_out3, pool_out4 = backbone.outputs
    # Change all to 256 units
    pyr_out1 = layers.Conv2D(256, 1, name='pyr_out1_conv1')(pool_out1)
    pyr_out2 = layers.Conv2D(256, 1, name='pyr_out2_conv1')(pool_out2)
    pyr_out3 = layers.Conv2D(256, 1, name='pyr_out3_conv1')(pool_out3)
    pyr_out4 = layers.Conv2D(256, 1, name='pyr_out4_conv1')(pool_out4)
    # pyramid handle
    pyr_out1, pyr_out2, pyr_out3, pyr_out4 = pyramid_block([pyr_out1, pyr_out2, pyr_out3, pyr_out4])
    # after pyramid
    pyr_out1 = layers.Conv2D(256, 3, 1, padding='same', name='pyr_out1_conv2')(pyr_out1)
    pyr_out2 = layers.Conv2D(256, 3, 1, padding='same', name='pyr_out2_conv2')(pyr_out2)
    pyr_out3 = layers.Conv2D(256, 3, 1, padding='same', name='pyr_out3_conv2')(pyr_out3)
    pyr_out4 = layers.Conv2D(256, 3, 1, padding='same', name='pyr_out4_conv2')(pyr_out4)
    # addition down sampling out5
    pyr_out5 = layers.Conv2D(256, 3, 1, padding='same', name='pyr_out5_conv')(pyr_out4)
    pyr_out5 = layers.BatchNormalization(epsilon=1.001e-5, name='pyr_out5_bn')(pyr_out5)
    pyr_out5 = layers.Activation('relu', name='pyr_out5_relu')(pyr_out5)
    pyr_out5 = layers.AveragePooling2D(name='pyr_out5_pool')(pyr_out5)
    # addition down sampling out6
    pyr_out6 = layers.Conv2D(256, 3, 1, padding='same', name='pyr_out6_conv')(pyr_out5)
    pyr_out6 = layers.BatchNormalization(epsilon=1.001e-5, name='pyr_out6_bn')(pyr_out6)
    pyr_out6 = layers.Activation('relu', name='pyr_out6_relu')(pyr_out6)
    pyr_out6 = layers.AveragePooling2D(name='pyr_out6_pool')(pyr_out6)
    return Model(inputs=[backbone.inputs],
                 outputs=[pyr_out1, pyr_out2, pyr_out3, pyr_out4, pyr_out5, pyr_out6])
