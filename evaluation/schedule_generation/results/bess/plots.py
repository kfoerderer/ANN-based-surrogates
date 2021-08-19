#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

presentation = True
matplotlib.rc('font', family='TeX Gyre Termes')

if presentation:
    matplotlib.rc('font', size=14)

#%%

quartiles = [
    ('random (4)', []),
    ('random (24)', [1.0, 1.0, 1.0, 1.0, 1.0]),
    ('random (96)', [1.0, 1.0, 1.0, 1.0, 1.0]),
    ('reference (4)', [1.0, 1.0, 1.0, 1.0, 1.0]),
    ('reference (24)', [1.0, 1.0, 1.0, 1.0, 1.0]),
    ('reference (96)', [1.0, 1.0, 1.0, 1.0, 2.0]),
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
    fig.savefig('results_bess_errors_per_schedule.png', dpi=300)
else:
    fig.savefig('results_bess_errors_per_schedule.pdf')

#%%


data = [
    ('random (4)', []),
    ('random (24)', [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]),
    ('random (96)', [-50.0, -50.0, -50.0, -50.0, -50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]),
    ('reference (4)', [-50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, 25.0, 37.5, 45.000000000000014, 50.0]),
    ('reference (24)', [-50.0, -50.0, -50.0, -50.0, -50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]),
    ('reference (96)', [-50.0, -50.0, -50.0, -50.0, -50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]),
]

quantiles = [0, 0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99, 1]
idx0, idx25, idx50, idx75, idx100 = quantiles.index(0), quantiles.index(0.25), quantiles.index(0.50), quantiles.index(0.75), quantiles.index(1)

quartiles = []
for (key, values) in data:
    if len(values) > 0:
        quartiles.append((key, [values[idx0], values[idx25], values[idx50], values[idx75], values[idx100]]))
    else:
        quartiles.append((key, []))


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
    fig.savefig('results_bess_errors.png', dpi=300)
else:
    fig.savefig('results_bess_errors.pdf')

# %%

x = np.arange(0,96+1)
series = [
    ('random', [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 9999, 9999, 9999, 9999, 9999, 9998, 9997, 9997, 9997, 9997, 9997, 9997, 9995, 9995, 9995, 9993, 9993, 9993, 9993, 9993, 9993, 9993, 9992, 9992, 9991, 9988, 9988, 9988, 9988, 9988, 9988, 9988, 9988, 9987, 9987, 9985, 9985, 9984, 9983, 9982, 9981, 9981, 9981, 9981, 9981, 9981, 9981, 9981, 9981, 9981, 9980, 9978, 9978, 9975, 9975, 9973, 9972, 9972, 9972, 9972, 9972, 9972, 9972, 9971, 9971, 9970, 9969, 9969, 9969, 9969, 9969, 9968, 9968, 9968, 9967, 9967, 9967, 9966, 9966, 9966, 9966, 9965, 9965, 9964, 9964, 9961, 9961]),
    ('random, 1% buffer', [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]),
    ('reference', [10000, 10000, 9997, 9995, 9994, 9993, 9989, 9987, 9982, 9979, 9978, 9976, 9969, 9966, 9961, 9951, 9943, 9938, 9928, 9914, 9903, 9892, 9880, 9870, 9865, 9857, 9850, 9837, 9823, 9812, 9797, 9784, 9770, 9756, 9750, 9735, 9718, 9710, 9694, 9678, 9662, 9648, 9634, 9612, 9591, 9588, 9572, 9562, 9543, 9529, 9513, 9489, 9471, 9462, 9456, 9435, 9413, 9397, 9381, 9366, 9340, 9329, 9310, 9290, 9262, 9251, 9232, 9216, 9189, 9177, 9158, 9140, 9118, 9106, 9091, 9076, 9047, 9032, 9017, 8995, 8958, 8946, 8932, 8903, 8875, 8864, 8850, 8835, 8820, 8805, 8790, 8771, 8751, 8735, 8719, 8698, 8676]),
    ('reference, 1% buffer', [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000])
]

normalization = 100/10000

colors = {
    'random': 'firebrick',
    'random, 1% buffer': 'sandybrown',
    'reference': 'darkblue',
    'reference, 1% buffer': 'dodgerblue',
}

fmt = {
    'random' : '-',
    'random, 1% buffer': '-',
    'reference': '-.',
    'reference, 1% buffer': '-.',
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
    fig.savefig('results_bess_feasibility.png', dpi=300)
else:
    fig.savefig('results_bess_feasibility.pdf')

# %%
