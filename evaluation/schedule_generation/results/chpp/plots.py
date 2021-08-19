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
    ('random (4)', [1.0, 2.0, 2.0, 2.0, 3.0]),
    ('random (24)', [1.0, 2.0, 2.0, 2.0, 12.0]),
    ('random (96)', [1.0, 2.0, 4.0, 4.0, 18.0]),
    ('reference (4)', [1.0, 2.0, 2.0, 2.0, 3.0]),
    ('reference (24)', [1.0, 2.0, 2.0, 4.0, 14.0]),
    ('reference (96)', [1.0, 2.0, 4.0, 4.0, 23.0]),
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
    fig.savefig('results_chpp_errors_per_schedule.png', dpi=300)
else:
    fig.savefig('results_chpp_errors_per_schedule.pdf')

#%%


data = [
    ('random (4)', [-2750.0, -2750.0, -2750.0, -2750.0, -2750.0, -2750.0, -1375.0, 5500.0, 5500.0, 5500.0, 5500.0]),
    ('random (24)', [-5500.0, -5500.0, -2750.0, -2750.0, -2750.0, -2750.0, 2750.0, 2750.0, 2750.0, 5500.0, 5500.0]),
    ('random (96)', [-5500.0, -5500.0, -5500.0, -2750.0, -2750.0, -2750.0, 2750.0, 2750.0, 5500.0, 5500.0, 5500.0]),
    ('reference (4)', [-5500.0, -4482.5, -2956.25, -2750.0, -2750.0, 2750.0, 2750.0, 5500.0, 5500.0, 5500.0, 5500.0]),
    ('reference (24)', [-5500.0, -5500.0, -5500.0, -5500.0, -2750.0, -2750.0, 2750.0, 5500.0, 5500.0, 5500.0, 5500.0]),
    ('reference (96)', [-5500.0, -5500.0, -5500.0, -2750.0, -2750.0, -2750.0, 2750.0, 2750.0, 5500.0, 5500.0, 5500.0]),
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
    fig.savefig('results_chpp_errors.png', dpi=300)
else:
    fig.savefig('results_chpp_errors.pdf')

# %%

x = np.arange(0,96+1)
series = [
    ('random', [10000, 10000, 9996, 9995, 9994, 9988, 9985, 9983, 9980, 9977, 9971, 9969, 9967, 9964, 9962, 9953, 9944, 9941, 9935, 9928, 9920, 9911, 9902, 9899, 9895, 9886, 9879, 9872, 9863, 9855, 9848, 9840, 9832, 9821, 9814, 9803, 9794, 9785, 9773, 9762, 9758, 9743, 9734, 9731, 9725, 9720, 9706, 9695, 9683, 9675, 9664, 9654, 9640, 9632, 9620, 9604, 9594, 9576, 9568, 9555, 9549, 9536, 9526, 9514, 9503, 9485, 9467, 9463, 9445, 9432, 9416, 9404, 9390, 9373, 9359, 9346, 9332, 9314, 9303, 9284, 9267, 9245, 9230, 9216, 9194, 9176, 9158, 9148, 9128, 9109, 9092, 9075, 9049, 9037, 9028, 9017, 9008]),
    ('random, 1°C buffer', [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]),
    ('reference', [10000, 9999, 9992, 9985, 9981, 9979, 9975, 9972, 9968, 9963, 9956, 9949, 9946, 9945, 9938, 9936, 9929, 9928, 9925, 9921, 9915, 9910, 9907, 9901, 9898, 9893, 9887, 9881, 9871, 9865, 9855, 9849, 9839, 9831, 9813, 9800, 9794, 9784, 9773, 9763, 9755, 9747, 9732, 9720, 9715, 9711, 9700, 9686, 9674, 9660, 9650, 9639, 9629, 9613, 9594, 9579, 9568, 9560, 9546, 9531, 9516, 9507, 9486, 9473, 9457, 9445, 9421, 9402, 9393, 9378, 9359, 9342, 9324, 9308, 9293, 9274, 9265, 9255, 9234, 9215, 9205, 9188, 9167, 9149, 9130, 9110, 9090, 9072, 9047, 9029, 9005, 8987, 8971, 8955, 8932, 8916, 8905]),
    ('reference, 1°C buffer', [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000])
]

normalization = 100/10000

colors = {
    'random': 'firebrick',
    'random, 1°C buffer': 'sandybrown',
    'reference': 'darkblue',
    'reference, 1°C buffer': 'dodgerblue',
}

fmt = {
    'random' : '-',
    'random, 1°C buffer': '-',
    'reference': '-.',
    'reference, 1°C buffer': '-.',
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
    fig.savefig('results_chpp_feasibility.png', dpi=300)
else:
    fig.savefig('results_chpp_feasibility.pdf')

# %%
