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
    ('random (4)', [1.0, 1.0, 1.0, 2.0, 4.0]),
    ('random (24)', [1.0, 1.0, 1.0, 2.0, 11.0]),
    ('random (96)', [1.0, 1.0, 1.0, 2.0, 12.0]),
    ('reference (4)', [1.0, 1.0, 1.0, 2.0, 4.0]),
    ('reference (24)', [1.0, 1.0, 1.0, 2.0, 11.0]),
    ('reference (96)', [1.0, 1.0, 1.0, 2.0, 15.0]),
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
    fig.savefig('results_evse_errors_per_schedule_a.png', dpi=300)
else:
    fig.savefig('results_evse_errors_per_schedule_a.pdf')


quartiles = [
    ('random (4),\n2% buffer', [1.0, 1.0, 1.0, 1.0, 1.0]),
    ('random (24),\n2% buffer', [1.0, 1.0, 2.0, 3.25, 7.0]),
    ('random (96),\n2% buffer', [1.0, 1.0, 1.0, 1.0, 7.0]),
    ('reference (4),\n2% buffer', [2.0, 2.0, 2.0, 2.0, 2.0]),
    ('reference (24),\n2% buffer', [1.0, 1.75, 2.0, 3.75, 7.0]),
    ('reference (96),\n2% buffer', [1.0, 1.0, 1.0, 1.0, 7.0]),
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
    fig.savefig('results_evse_errors_per_schedule_b.png', dpi=300)
else:
    fig.savefig('results_evse_errors_per_schedule_b.pdf')

#%%


data = [
    ('random (4)', [-4840.0, -3740.0, -2640.0, -1980.0, -660.0, -440.0, -220.0, 440.0, 3740.0, 3740.0, 3960.0]),
    ('random (24)', [-4840.0, -2640.0, -1760.0, -1100.0, -440.0, -220.0, 440.0, 3960.0, 3960.0, 4180.0, 5720.0]),
    ('random (96)', [-4840.0, -1980.0, -1320.0, -880.0, -220.0, 440.0, 3740.0, 3960.0, 4180.0, 4400.0, 7920.0]),
    ('reference (4)', [-5500.0, -3667.4, -2860.0, -2200.0, -660.0, -440.0, -220.0, 220.0, 588.49999999999, 3740.0, 3740.0]),
    ('reference (24)', [-5500.0, -2640.0, -1760.0, -1100.0, -440.0, -220.0, 220.0, 3740.0, 3960.0, 4180.0, 5940.0]),
    ('reference (96)', [-5500.0, -2200.0, -1320.0, -880.0, -220.0, 440.0, 3740.0, 3960.0, 3960.0, 4180.0, 6820.0]),
]

quantiles = [0, 0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99, 1]
idx0, idx25, idx50, idx75, idx100 = quantiles.index(0), quantiles.index(0.25), quantiles.index(0.50), quantiles.index(0.75), quantiles.index(1)

quartiles = []
for (key, values) in data:
    if len(values) > 0:
        quartiles.append((key, [values[idx0], values[idx25], values[idx50], values[idx75], values[idx100]]))
    else:
        quartiles.append((key,()))


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
    fig.savefig('results_evse_errors_a.png', dpi=300)
else:
    fig.savefig('results_evse_errors_a.pdf')


data = [
    ('random (4),\n2% buffer', [-1100.0, -1100.0, -1100.0, -1100.0, -1100.0, -1100.0, -1100.0, -1100.0, -1100.0, -1100.0, -1100.0]),
    ('random (24),\n2% buffer',  [-1320.0, -1276.0, -1210.0, -1100.0, -440.0, -220.0, -220.0, 3740.0, 3740.0, 3740.0, 3740.0]),
    ('random (96),\n2% buffer', [-1320.0, -1141.8, -874.5, -440.0, 220.0, 3740.0, 3740.0, 3960.0, 4174.499999999999, 4221.799999999999, 4400.0]),
    ('reference (4),\n2% buffer', [-880.0, -875.6, -869.0, -858.0, -770.0, -660.0, -550.0, -462.00000000000006, -451.0, -444.40000000000003, -440.0]),
    ('reference (24),\n2% buffer', [-880.0, -829.4000000000001, -753.5, -660.0, -440.0, -220.0, -220.0, -220.0, 1463.000000000003, 2829.1999999999985, 3740.0]),
    ('reference (96),\n2% buffer', [-880.0, -715.0, -660.0, -440.0, -220.0, 3740.0, 3795.0, 3960.0, 4180.0, 4235.0, 4400.0]),
]

quantiles = [0, 0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99, 1]
idx0, idx25, idx50, idx75, idx100 = quantiles.index(0), quantiles.index(0.25), quantiles.index(0.50), quantiles.index(0.75), quantiles.index(1)

quartiles = []
for (key, values) in data:
    if len(values) > 0:
        quartiles.append((key, [values[idx0], values[idx25], values[idx50], values[idx75], values[idx100]]))
    else:
        quartiles.append((key,()))


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
    fig.savefig('results_evse_errors_b.png', dpi=300)
else:
    fig.savefig('results_evse_errors_b.pdf')

# %%

x = np.arange(0,96+1)
series = [
    ('random', [10000, 9997, 9869, 9659, 9439, 9216, 9080, 8980, 8891, 8827, 8766, 8720, 8646, 8585, 8523, 8463, 8414, 8360, 8315, 8263, 8198, 8150, 8107, 8068, 8022, 7985, 7945, 7904, 7869, 7834, 7801, 7766, 7733, 7699, 7671, 7641, 7615, 7585, 7563, 7530, 7502, 7472, 7436, 7403, 7374, 7339, 7311, 7280, 7253, 7216, 7192, 7160, 7134, 7108, 7081, 7043, 7012, 6978, 6961, 6939, 6907, 6880, 6854, 6820, 6786, 6755, 6719, 6698, 6665, 6635, 6613, 6586, 6551, 6527, 6492, 6471, 6431, 6400, 6370, 6339, 6314, 6280, 6245, 6212, 6188, 6162, 6129, 6096, 6062, 6031, 6012, 5981, 5957, 5931, 5903, 5859, 5833]),
    ('random, 1% buffer', [10000, 10000, 9998, 9992, 9986, 9974, 9966, 9950, 9934, 9924, 9915, 9912, 9909, 9906, 9906, 9904, 9903, 9898, 9894, 9890, 9885, 9883, 9883, 9880, 9876, 9872, 9869, 9868, 9863, 9858, 9854, 9853, 9852, 9850, 9847, 9843, 9841, 9836, 9832, 9823, 9820, 9812, 9806, 9803, 9793, 9791, 9787, 9781, 9775, 9771, 9767, 9761, 9754, 9751, 9747, 9741, 9736, 9728, 9718, 9707, 9697, 9691, 9683, 9676, 9673, 9669, 9662, 9652, 9645, 9639, 9635, 9634, 9628, 9622, 9616, 9612, 9599, 9591, 9587, 9577, 9572, 9566, 9556, 9547, 9540, 9534, 9526, 9522, 9513, 9503, 9494, 9485, 9479, 9473, 9465, 9461, 9454]),
    ('random, 2% buffer', [10000, 10000, 10000, 10000, 9999, 9999, 9999, 9998, 9997, 9996, 9995, 9994, 9994, 9994, 9994, 9994, 9994, 9994, 9994, 9994, 9994, 9994, 9994, 9993, 9992, 9992, 9992, 9992, 9992, 9992, 9992, 9992, 9992, 9992, 9992, 9992, 9992, 9992, 9992, 9991, 9990, 9990, 9990, 9990, 9990, 9990, 9990, 9990, 9990, 9989, 9989, 9989, 9989, 9989, 9989, 9989, 9988, 9988, 9988, 9987, 9986, 9986, 9985, 9985, 9984, 9984, 9981, 9980, 9980, 9979, 9979, 9977, 9975, 9973, 9973, 9969, 9966, 9965, 9962, 9961, 9960, 9958, 9954, 9952, 9951, 9951, 9950, 9949, 9948, 9948, 9946, 9945, 9943, 9942, 9941, 9940, 9937]),
    ('reference', [10000, 9986, 9769, 9521, 9335, 9153, 8990, 8895, 8823, 8760, 8693, 8634, 8574, 8518, 8453, 8388, 8338, 8277, 8234, 8175, 8111, 8056, 8013, 7979, 7942, 7896, 7852, 7800, 7760, 7728, 7684, 7634, 7593, 7553, 7520, 7495, 7459, 7428, 7393, 7369, 7346, 7317, 7289, 7260, 7235, 7209, 7178, 7143, 7115, 7087, 7060, 7034, 6999, 6968, 6942, 6924, 6894, 6865, 6841, 6812, 6783, 6756, 6718, 6694, 6667, 6635, 6600, 6568, 6536, 6509, 6485, 6460, 6436, 6416, 6388, 6363, 6343, 6319, 6291, 6265, 6234, 6193, 6165, 6138, 6110, 6078, 6043, 6010, 5979, 5940, 5912, 5877, 5850, 5819, 5781, 5740, 5710]),
    ('reference, 1% buffer', [10000, 10000, 9995, 9988, 9982, 9970, 9960, 9946, 9940, 9929, 9921, 9918, 9915, 9912, 9908, 9907, 9903, 9901, 9899, 9897, 9897, 9896, 9895, 9894, 9893, 9892, 9889, 9883, 9882, 9880, 9873, 9869, 9867, 9864, 9859, 9851, 9847, 9844, 9840, 9835, 9832, 9829, 9826, 9820, 9819, 9814, 9808, 9807, 9805, 9801, 9792, 9784, 9777, 9771, 9764, 9761, 9756, 9748, 9739, 9730, 9725, 9719, 9714, 9704, 9699, 9688, 9684, 9679, 9675, 9668, 9658, 9648, 9639, 9632, 9625, 9618, 9609, 9602, 9596, 9590, 9581, 9578, 9568, 9560, 9549, 9543, 9536, 9529, 9520, 9518, 9517, 9510, 9502, 9496, 9488, 9483, 9479]),
    ('reference, 2% buffer', [10000, 10000, 9999, 9999, 9999, 9997, 9997, 9997, 9996, 9995, 9994, 9994, 9993, 9993, 9993, 9992, 9992, 9992, 9992, 9992, 9992, 9992, 9992, 9992, 9992, 9991, 9991, 9991, 9991, 9991, 9991, 9991, 9991, 9991, 9991, 9991, 9991, 9991, 9991, 9990, 9990, 9990, 9990, 9990, 9989, 9989, 9989, 9989, 9989, 9988, 9987, 9987, 9986, 9986, 9986, 9985, 9985, 9985, 9985, 9985, 9984, 9984, 9984, 9983, 9982, 9980, 9980, 9979, 9977, 9976, 9975, 9971, 9970, 9969, 9968, 9965, 9963, 9963, 9963, 9962, 9961, 9959, 9957, 9957, 9956, 9955, 9954, 9953, 9952, 9950, 9949, 9948, 9948, 9946, 9944, 9944, 9943])
]

normalization = 100/10000

colors = {
    'random': 'firebrick',
    'random, 1% buffer': 'sandybrown',
    'random, 2% buffer': 'gold',
    'reference': 'darkblue',
    'reference, 1% buffer': 'dodgerblue',
    'reference, 2% buffer': 'teal',
}

fmt = {
    'random' : '-',
    'random, 1% buffer': '-',
    'random, 2% buffer': '-',
    'reference': '-.',
    'reference, 1% buffer': '-.',
    'reference, 2% buffer': '-.',
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
    fig.savefig('results_evse_feasibility.png', dpi=300)
else:
    fig.savefig('results_evse_feasibility.pdf')

# %%
