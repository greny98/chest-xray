import tensorflow as tf

tensor = tf.convert_to_tensor(
    [0.06757758, 0.01095575, 0.07097605, 0.00114477, 0.15416798, 0.08357637, 0.04039592, 0.15397441, 0.02315011,
     0.01737617, 0.00711973, 0.0104767, 0.00962774, 0.07134485])
max_value = tf.reduce_max(tensor)
print(tensor/max_value)
