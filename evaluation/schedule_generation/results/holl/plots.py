#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

presentation = False
matplotlib.rc('font', family='TeX Gyre Termes')

if presentation:
    matplotlib.rc('font', size=14)
#%%

quartiles = [
    ('random (4)', [1.0, 1.0, 1.0, 1.0, 3.0]),
    ('random (24)', [1.0, 1.0, 1.0, 2.0, 14.0]),
    ('random (96)', [1.0, 1.0, 3.0, 5.0, 54.0]),
    ('reference (4)', [1.0, 1.0, 1.0, 1.0, 4.0]),
    ('reference (24)', [1.0, 1.0, 1.0, 2.0, 19.0]),
    ('reference (96)', [1.0, 2.0, 4.0, 8.0, 65.0]),
]

fig, ax = plt.subplots(figsize=(8,4))

for idx, (label, values) in enumerate(quartiles):
    boxplot = ax.boxplot(values, whis=(0,100), positions=[idx])#, medianprops=medianprops)

    for median in boxplot['medians']:
        median.set(color='firebrick', linewidth=1.5,)
        x,y = median.get_data()
        xn = (x-(x.sum()/2.))*0.5+(x.sum()/2.)
        plt.plot(xn, y, color="firebrick", linewidth=7, solid_capstyle="butt", zorder=4)

if presentation:
    plt.title('Number of errors per (infeasible) load schedule')
else:
    ax.set(ylabel='Number of errors per (infeasible) load schedule')
ax.set(xticklabels = [label for (label, values) in quartiles])
fig.tight_layout()
plt.show()

if presentation:
    fig.savefig('results_holl_errors_per_schedule.png', dpi=300)
else:
    fig.savefig('results_holl_errors_per_schedule.pdf')

#%%


data = [
    ('random (4)', [-10900.0, -6923.999999999999, -5500.0, -3859.9999999999995, -1000.0, -400.0, -100.0, 2974.9999999999886, 4114.999999999998, 5200.0, 5500.0]),
    ('random (24)', [-10900.0, -5500.0, -4700.0, -3450.0, -1200.0, -200.0, 1000.0, 3600.0, 4800.0, 5400.0, 13300.0]),
    ('random (96)', [-10900.0, -5300.0, -4600.0, -3594.999999999993, -1500.0, -200.0, 1400.0, 3700.0, 4700.0, 5300.0, 23400.0]),
    ('reference (4)', [-14700.0, -6000.0, -4500.0, -3495.0, -1000.0, -300.0, -100.0, 1794.9999999999989, 2700.0, 4398.000000000002, 5500.0]),
    ('reference (24)', [-14700.0, -5100.0, -4100.0, -2600.0, -300.0, 100.0, 300.0, 3500.0, 4400.0, 5400.0, 14600.0]),
    ('reference (96)', [-17800.0, -4500.0, -2900.0, -2200.0, 100.0, 200.0, 300.0, 3400.0, 4700.0, 5800.0, 21500.0]),
]

quantiles = [0, 0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99, 1]
idx0, idx25, idx50, idx75, idx100 = quantiles.index(0), quantiles.index(0.25), quantiles.index(0.50), quantiles.index(0.75), quantiles.index(1)

quartiles = []
for (key, values) in data:
    quartiles.append((key, [values[idx0], values[idx25], values[idx50], values[idx75], values[idx100]]))


fig, ax = plt.subplots(figsize=(8,4))

for idx, (label, values) in enumerate(quartiles):
    boxplot = ax.boxplot(values, whis=(0,100), positions=[idx])#, medianprops=medianprops)

    for median in boxplot['medians']:
        median.set(color='firebrick', linewidth=1.5,)
        x,y = median.get_data()
        xn = (x-(x.sum()/2.))*0.5+(x.sum()/2.)
        plt.plot(xn, y, color="firebrick", linewidth=7, solid_capstyle="butt", zorder=4)

if presentation:
    plt.title('Errors (unequal 0) in W')
else:
    ax.set(ylabel='Errors (unequal 0) in W')
ax.set(xticklabels = [label for (label, values) in quartiles])
fig.tight_layout()
plt.show()

if presentation:
    fig.savefig('results_holl_errors.png', dpi=300)
else:
    fig.savefig('results_holl_errors.pdf')

# %%

x = np.arange(0,96+1)
series = [
    ('random, 1% + 1°C buffer', [10000, 10000, 9981, 9960, 9926, 9901, 9882, 9857, 9834, 9821, 9811, 9801, 9794, 9784, 9767, 9752, 9740, 9728, 9714, 9697, 9678, 9670, 9653, 9639, 9623, 9608, 9596, 9581, 9557, 9540, 9520, 9502, 9479, 9462, 9446, 9419, 9395, 9372, 9358, 9344, 9321, 9292, 9274, 9249, 9224, 9203, 9173, 9141, 9115, 9084, 9055, 9026, 8997, 8970, 8944, 8909, 8878, 8834, 8801, 8771, 8729, 8694, 8660, 8629, 8596, 8559, 8536, 8497, 8467, 8435, 8399, 8367, 8329, 8293, 8258, 8232, 8184, 8149, 8111, 8084, 8054, 8014, 7976, 7935, 7901, 7856, 7815, 7784, 7740, 7705, 7661, 7634, 7591, 7551, 7523, 7480, 7439]),
    ('random*, 1% + 1°C buffer', [10000, 10000, 9996, 9995, 9992, 9989, 9987, 9983, 9980, 9976, 9974, 9973, 9972, 9971, 9964, 9959, 9957, 9956, 9955, 9953, 9952, 9951, 9949, 9948, 9947, 9943, 9942, 9939, 9936, 9934, 9931, 9927, 9922, 9920, 9918, 9915, 9911, 9909, 9908, 9907, 9904, 9900, 9897, 9894, 9891, 9890, 9888, 9883, 9881, 9876, 9873, 9871, 9868, 9862, 9856, 9851, 9846, 9841, 9838, 9835, 9832, 9831, 9830, 9829, 9826, 9826, 9825, 9821, 9819, 9815, 9813, 9810, 9807, 9800, 9797, 9795, 9792, 9791, 9787, 9787, 9786, 9784, 9784, 9783, 9782, 9780, 9777, 9777, 9777, 9775, 9773, 9773, 9772, 9772, 9772, 9771, 9769]),
    ('random, 2% + 2°C buffer', [10000, 10000, 9986, 9969, 9950, 9929, 9900, 9887, 9874, 9854, 9836, 9824, 9813, 9807, 9790, 9766, 9754, 9739, 9724, 9704, 9685, 9671, 9657, 9648, 9636, 9625, 9604, 9582, 9561, 9551, 9530, 9516, 9498, 9472, 9456, 9438, 9418, 9401, 9377, 9343, 9317, 9291, 9262, 9233, 9204, 9186, 9165, 9145, 9111, 9080, 9062, 9030, 9008, 8974, 8952, 8920, 8893, 8867, 8836, 8808, 8789, 8757, 8718, 8681, 8648, 8616, 8588, 8560, 8523, 8495, 8460, 8435, 8408, 8371, 8337, 8293, 8268, 8239, 8211, 8185, 8153, 8122, 8095, 8059, 8027, 8000, 7955, 7910, 7879, 7835, 7806, 7776, 7741, 7711, 7674, 7637, 7608]),
    ('random*, 2% + 2°C buffer', [10000, 10000, 10000, 9999, 9999, 9999, 9997, 9997, 9997, 9995, 9992, 9990, 9986, 9986, 9986, 9984, 9983, 9979, 9978, 9976, 9974, 9971, 9971, 9971, 9969, 9969, 9968, 9966, 9965, 9965, 9965, 9963, 9962, 9960, 9959, 9958, 9956, 9956, 9956, 9953, 9949, 9946, 9945, 9944, 9942, 9941, 9939, 9939, 9939, 9936, 9936, 9936, 9935, 9931, 9930, 9928, 9925, 9923, 9922, 9922, 9921, 9918, 9917, 9917, 9915, 9914, 9913, 9913, 9911, 9909, 9907, 9905, 9905, 9905, 9905, 9903, 9903, 9901, 9900, 9900, 9900, 9897, 9897, 9896, 9894, 9894, 9894, 9893, 9893, 9891, 9891, 9891, 9890, 9889, 9889, 9888, 9887]),
    ('reference, 1% + 1°C buffer', [10000, 10000, 9953, 9911, 9885, 9856, 9832, 9816, 9804, 9794, 9778, 9768, 9750, 9739, 9728, 9710, 9698, 9686, 9666, 9648, 9634, 9612, 9588, 9568, 9555, 9532, 9507, 9487, 9464, 9446, 9425, 9396, 9376, 9351, 9320, 9297, 9265, 9244, 9219, 9182, 9153, 9123, 9098, 9075, 9045, 9012, 8981, 8944, 8912, 8885, 8857, 8827, 8786, 8753, 8718, 8677, 8633, 8596, 8560, 8511, 8479, 8434, 8400, 8352, 8311, 8280, 8238, 8191, 8147, 8108, 8064, 8024, 7974, 7931, 7883, 7833, 7785, 7742, 7688, 7623, 7566, 7521, 7474, 7428, 7377, 7341, 7287, 7233, 7185, 7143, 7091, 7039, 6985, 6950, 6893, 6832, 6779]),
    ('reference*, 1% + 1°C buffer', [10000, 10000, 9985, 9978, 9976, 9971, 9968, 9966, 9961, 9960, 9958, 9955, 9952, 9951, 9948, 9947, 9942, 9941, 9938, 9936, 9934, 9931, 9926, 9922, 9916, 9913, 9909, 9906, 9903, 9898, 9895, 9894, 9892, 9887, 9882, 9878, 9872, 9871, 9865, 9862, 9856, 9853, 9849, 9848, 9842, 9836, 9830, 9825, 9820, 9819, 9818, 9813, 9807, 9803, 9800, 9793, 9790, 9784, 9781, 9774, 9769, 9763, 9761, 9754, 9753, 9747, 9745, 9740, 9737, 9732, 9731, 9730, 9727, 9719, 9714, 9711, 9708, 9706, 9701, 9698, 9693, 9690, 9685, 9684, 9680, 9680, 9676, 9673, 9669, 9665, 9659, 9657, 9654, 9651, 9647, 9644, 9643]),
    ('reference, 2% + 2°C buffer', [10000, 10000, 9972, 9926, 9891, 9865, 9840, 9821, 9797, 9784, 9775, 9761, 9751, 9741, 9725, 9709, 9689, 9672, 9663, 9648, 9633, 9615, 9593, 9575, 9555, 9541, 9515, 9489, 9469, 9440, 9418, 9385, 9365, 9345, 9323, 9299, 9273, 9239, 9208, 9179, 9157, 9124, 9094, 9070, 9039, 9015, 8993, 8954, 8929, 8900, 8869, 8828, 8804, 8780, 8742, 8710, 8678, 8647, 8604, 8564, 8532, 8495, 8458, 8408, 8374, 8343, 8305, 8268, 8233, 8195, 8150, 8108, 8060, 8003, 7965, 7933, 7892, 7843, 7811, 7775, 7729, 7687, 7643, 7618, 7576, 7521, 7473, 7427, 7392, 7346, 7290, 7249, 7198, 7147, 7112, 7063, 7013]),
    ('reference*, 2% + 2°C buffer', [10000, 10000, 9998, 9995, 9994, 9993, 9992, 9991, 9990, 9989, 9988, 9987, 9987, 9987, 9987, 9985, 9984, 9984, 9984, 9982, 9982, 9982, 9982, 9980, 9980, 9978, 9977, 9974, 9971, 9970, 9970, 9968, 9967, 9964, 9962, 9962, 9959, 9953, 9952, 9950, 9949, 9947, 9947, 9945, 9942, 9942, 9942, 9938, 9937, 9935, 9935, 9935, 9934, 9933, 9932, 9930, 9929, 9926, 9924, 9922, 9920, 9919, 9917, 9915, 9913, 9912, 9910, 9909, 9908, 9907, 9904, 9902, 9899, 9895, 9895, 9895, 9894, 9891, 9890, 9890, 9890, 9889, 9888, 9888, 9886, 9886, 9885, 9883, 9882, 9881, 9879, 9877, 9873, 9873, 9872, 9872, 9872]),
]

normalization = 100/10000

colors = {
    'random, 1% + 1°C buffer': 'sandybrown',
    'random*, 1% + 1°C buffer': 'sandybrown',
    'random, 2% + 2°C buffer': 'gold',
    'random*, 2% + 2°C buffer': 'gold',
    'reference, 1% + 1°C buffer': 'dodgerblue',
    'reference*, 1% + 1°C buffer': 'dodgerblue',
    'reference, 2% + 2°C buffer': 'teal',
    'reference*, 2% + 2°C buffer': 'teal',
}

fmt = {
    'random, 1% + 1°C buffer': '-',
    'random*, 1% + 1°C buffer': '--',
    'random, 2% + 2°C buffer': '-',
    'random*, 2% + 2°C buffer': '--',
    'reference, 1% + 1°C buffer': '-.',
    'reference*, 1% + 1°C buffer': ':',
    'reference, 2% + 2°C buffer': '-.',
    'reference*, 2% + 2°C buffer': ':',
}

fig, ax = plt.subplots(figsize=(8,4))

for label, values in series:
    ax.plot(x, np.array(values)*normalization, fmt[label], label=label, color=colors[label])

ax.legend(loc='lower left')
ax.legend(frameon=False)
#ax.set(xticklabels = [label for (label, values) in quartiles])
#ax.set(yticks=ticks)
ax.set(xlabel='Time step')
if presentation:
    plt.title('Percentage of feasible load schedules')
else:
    ax.set(ylabel='Percentage of feasible load schedules')
fig.tight_layout()
plt.show()

if presentation:
    fig.savefig('results_holl_feasibility_b.png', dpi=300)
else:
    fig.savefig('results_holl_feasibility_b.pdf')

# %%

x = np.arange(0,96+1)
series = [
    ('random', [10000, 9984, 9917, 9826, 9739, 9655, 9598, 9560, 9524, 9492, 9464, 9427, 9392, 9372, 9340, 9313, 9282, 9251, 9217, 9187, 9150, 9112, 9064, 9031, 8996, 8963, 8924, 8879, 8841, 8797, 8760, 8721, 8663, 8619, 8572, 8524, 8477, 8431, 8374, 8320, 8264, 8220, 8169, 8114, 8067, 8021, 7962, 7909, 7843, 7789, 7721, 7668, 7618, 7546, 7495, 7442, 7380, 7313, 7251, 7194, 7130, 7067, 7019, 6973, 6905, 6835, 6780, 6731, 6670, 6609, 6559, 6504, 6457, 6401, 6343, 6272, 6214, 6162, 6112, 6066, 6021, 5963, 5911, 5842, 5782, 5718, 5659, 5599, 5553, 5514, 5453, 5403, 5346, 5302, 5259, 5214, 5167]),
    ('random*', [10000, 9984, 9941, 9899, 9862, 9835, 9813, 9797, 9783, 9777, 9768, 9754, 9741, 9727, 9717, 9698, 9679, 9669, 9655, 9643, 9624, 9600, 9572, 9555, 9531, 9516, 9495, 9467, 9441, 9417, 9397, 9372, 9337, 9309, 9286, 9261, 9232, 9203, 9169, 9142, 9113, 9093, 9070, 9041, 9018, 8990, 8961, 8930, 8892, 8859, 8827, 8803, 8775, 8737, 8709, 8678, 8652, 8613, 8583, 8548, 8518, 8491, 8471, 8449, 8422, 8389, 8366, 8337, 8315, 8276, 8254, 8235, 8210, 8187, 8157, 8124, 8096, 8069, 8047, 8031, 8008, 7987, 7970, 7943, 7928, 7903, 7883, 7864, 7851, 7839, 7827, 7807, 7787, 7770, 7760, 7751, 7740]),
    ('reference', [10000, 9946, 9788, 9671, 9581, 9488, 9407, 9337, 9278, 9231, 9181, 9135, 9085, 9047, 8991, 8941, 8888, 8836, 8780, 8725, 8680, 8631, 8571, 8512, 8458, 8405, 8345, 8261, 8211, 8147, 8079, 8016, 7955, 7886, 7820, 7748, 7675, 7624, 7554, 7484, 7414, 7341, 7289, 7230, 7177, 7122, 7056, 7008, 6939, 6877, 6819, 6752, 6694, 6631, 6558, 6504, 6424, 6351, 6276, 6214, 6166, 6112, 6033, 5974, 5918, 5862, 5814, 5754, 5703, 5650, 5598, 5541, 5481, 5431, 5374, 5316, 5272, 5215, 5175, 5115, 5035, 4971, 4913, 4868, 4818, 4760, 4706, 4657, 4607, 4557, 4504, 4446, 4391, 4349, 4303, 4255, 4216]),
    ('reference*', [10000, 9946, 9853, 9811, 9775, 9735, 9699, 9672, 9648, 9627, 9603, 9588, 9563, 9543, 9519, 9492, 9470, 9450, 9425, 9400, 9385, 9360, 9337, 9308, 9280, 9258, 9239, 9199, 9178, 9153, 9120, 9097, 9070, 9043, 9020, 8995, 8966, 8944, 8916, 8889, 8864, 8828, 8809, 8788, 8764, 8741, 8708, 8689, 8669, 8651, 8629, 8608, 8588, 8564, 8538, 8512, 8490, 8462, 8435, 8412, 8402, 8383, 8353, 8331, 8316, 8298, 8285, 8265, 8255, 8241, 8224, 8208, 8190, 8174, 8157, 8147, 8135, 8117, 8105, 8091, 8069, 8053, 8040, 8027, 8014, 8000, 7982, 7968, 7958, 7953, 7938, 7924, 7912, 7906, 7894, 7885, 7882]),
]

normalization = 100/10000

colors = {
    'random': 'firebrick',
    'random*': 'firebrick',
    'reference': 'darkblue',
    'reference*': 'darkblue',
}

fmt = {
    'random' : '-',
    'random*' : '--',
    'reference': '-.',
    'reference*': ':',
}

fig, ax = plt.subplots(figsize=(8,4))

for label, values in series:
    ax.plot(x[:len(values)], np.array(values)*normalization, fmt[label], label=label, color=colors[label])
    

ax.legend(loc='lower left')
ax.legend(frameon=False)
#ax.set(xticklabels = [label for (label, values) in quartiles])
#ax.set(yticks=ticks)
ax.set(xlabel='Time step')
if presentation:
    plt.title('Percentage of feasible load schedules')
else:
    ax.set(ylabel='Percentage of feasible load schedules')
fig.tight_layout()
plt.show()

if presentation:
    fig.savefig('results_holl_feasibility_a.png', dpi=300)
else:
    fig.savefig('results_holl_feasibility_a.pdf')

# %%