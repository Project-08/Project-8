import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from numerics_results import data as fdm_data

problem = 'P5'

data = {
    'P1' : {'title' : r'P1: $\Delta u = \sin(\pi x) \sin(\pi y)$'},
    'P2' : {'title' : r'P2: $\Delta u = (x^2 + y^2) e^{xy}$'},
    'P3' : {'title' : r'P3: $\Delta u = 2(xy + xz + yz)$'},
    'P4' : {'title' : r'P4: $\Delta u = 6$'},
    'P5' : {'title' : r'P5: $\Delta u = -\pi^2 xy \sin(\pi z)$'},
    'P6' : {'title' : r'P6: $\Delta u = -3\pi^2 \sin(\pi x) \sin(\pi y) \sin(\pi z)$'},
}

# Load the data
with open('error_v_time_results.pkl', 'rb') as f:
    result_lists = pkl.load(f)


for i in range(1, 7):
    key = 'P' + str(i)
    key_drm = key + ' DRM'
    data[key]['drm'] = {
        'time' : result_lists[key_drm][1],
        'error' : result_lists[key_drm][0]
    }
    key_pinn = key + ' PINN'
    data[key]['pinn'] = {
        'time' : result_lists[key_pinn][1],
        'error' : result_lists[key_pinn][0]
    }
    if key in fdm_data:
        data[key]['fdm'] = {
            'time' : fdm_data[key]['fdm']['time'],
            'error' : fdm_data[key]['fdm']['error']
        }

if 'drm' in data[problem]:
    plt.plot(data[problem]['drm']['time'], data[problem]['drm']['error'], label='DRM', color='lightblue', linestyle='-', linewidth=2)
    plt.scatter(data[problem]['drm']['time'], data[problem]['drm']['error'], c='blue', marker='x', s=50, alpha=0.7)
if 'pinn' in data[problem]:
    plt.plot(data[problem]['pinn']['time'], data[problem]['pinn']['error'], label='PINN', color='lightgreen', linestyle='-', linewidth=2)
    plt.scatter(data[problem]['pinn']['time'], data[problem]['pinn']['error'], c='green', marker='x', s=50, alpha=0.7)
if 'fdm' in data[problem]:
    plt.plot(data[problem]['fdm']['time'], data[problem]['fdm']['error'], label='FDM', color='lightcoral', linestyle='-', linewidth=2)
    plt.scatter(data[problem]['fdm']['time'], data[problem]['fdm']['error'], c='red', marker='x', s=50, alpha=0.7)
plt.xlabel('Time [s]')
plt.ylabel('RMSE')
plt.yscale('log')
plt.xscale('log')
plt.title(data[problem]['title'])
plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(problem + '_comp.png')
plt.show()
