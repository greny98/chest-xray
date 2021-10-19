import numpy as np

# ======== y_true, y_pred tf.Tensor([0.3621515  0.42201838 0.19677839 0.48591319], shape=(4,), dtype=float64) tf.Tensor([1. 0. 0. 0.], shape=(4,), dtype=float64)
# ======== pos_loss, neg_loss, mean_loss tf.Tensor([-3.55008073e-08  6.66796101e+00  3.10913144e+00  7.67750968e+00], shape=(4,), dtype=float64) tf.Tensor([ 2.02790090e-01 -1.14006463e-09 -1.58434891e-09 -1.01403257e-09], shape=(4,), dtype=float64) tf.Tensor(4.414348045009684, shape=(), dtype=float64)

y_pred = np.array([0.7558375, 0.68601155, 0.63180149, 0.61579627])
y_true = np.array([0., 0., 0., 0.])
print(-(1 - y_true) * np.log(1 - y_pred + 1e-7))

# [11.94235899 10.83909734  9.98256921  9.72968409
