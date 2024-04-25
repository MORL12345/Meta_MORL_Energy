import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
agg_powers_no_renew = np.load("data/new_agg_no_renew_kWh_14_15.npy")
agg_powers_renew = np.load("data/new_agg_renew_kWh_14_15.npy")
excluded_powers_no_renew = np.load("data/excluded_power_no_renew_kWh_14_15.npy")
excluded_powers_renew = np.load("data/excluded_power_renew_kWh_14_15.npy")


def average_every_24(lst):
    result = []
    min_result = []
    max_result = []
    n = len(lst)
    for i in range(0, n, 24):
        chunk = lst[i:i + 24]
        avg = sum(chunk) / len(chunk)
        result.append(avg)
        min_result.append(min(chunk))
        max_result.append(max(chunk))
    return result, min_result, max_result


# avg_agg_power_no_renew, min_agg_power_no_renew, max_agg_power_no_renew = average_every_24(agg_powers_no_renew)
# plt.plot(range(1,len(avg_agg_power_no_renew)+1),avg_agg_power_no_renew, color="blue")
# plt.fill_between(range(1,len(avg_agg_power_no_renew)+1), max_agg_power_no_renew, min_agg_power_no_renew, color='blue', alpha=0.3)

# avg_excluded_power_no_renew, min_excluded_power_no_renew, max_excluded_power_no_renew = average_every_24(excluded_powers_no_renew)
# plt.plot(range(len(avg_excluded_power_no_renew)), avg_excluded_power_no_renew, color="red")
# plt.fill_between(range(len(avg_excluded_power_no_renew)), max_excluded_power_no_renew, min_excluded_power_no_renew, color='red', alpha=0.3)
#
# avg_agg_power_renew, min_agg_power_renew, max_agg_power_renew = average_every_24(agg_powers_renew)
# plt.plot(range(len(avg_agg_power_renew)), avg_agg_power_renew, color="green")
# plt.fill_between(range(len(avg_agg_power_renew)), max_agg_power_renew, min_agg_power_renew, color='green', alpha=0.3)
#
#
avg_excluded_power_renew, min_excluded_power_renew, max_excluded_power_renew = average_every_24(agg_powers_no_renew-agg_powers_renew)
plt.plot(range(len(avg_excluded_power_renew)), avg_excluded_power_renew, color="green", label="Renewable Generation")
plt.fill_between(range(len(avg_excluded_power_renew)), max_excluded_power_renew, min_excluded_power_renew, color='green', alpha=0.3)
# plt.xticks(range(365), range(1,366))
print(len(avg_excluded_power_renew))
for x in [1/7, 4, 6, 8, 10, 12, 16, 23, 29, 33, 38, 51]:
    plt.axvline(x=x*7, color='red', linestyle='--', linewidth=1)
    plt.text(int(x*7)-3, -0.22, f'{int(x*7)}', color='black', fontsize=10)
# plt.xticks([0, 365], ['0', '365'])
plt.xticks([])
plt.xlabel('Day', labelpad=20,fontsize=16)
plt.ylabel('Power (kW)',fontsize=16)
plt.title("Average Daily Renewable Generation Shifting Point")
legend_elements = [Line2D([0], [0], color='green', label='Renewable Generation'),
                   Line2D([0], [0], color='red', linestyle='--', label='Context Shifting Point (day)')]
plt.legend(handles=legend_elements, loc='upper right', fontsize=14)

plt.show()
