import numpy as np

boiler_hourly_data = np.load("data/boiler_hourly_power_kWh_14_15.npy")
agg_hourly_data = np.load("data/agg_hourly_power_kWh_14_15.npy")
solar_hourly_data = np.load("data/solar_pump_hourly_power_kWh_14_15.npy")
renewable_hourly_data = np.load("data/renewable_generation_kW_14_15_London.npy")
heater_power = 1.5  # kW

excluded_powers_renew = []
excluded_powers_no_renew = []
agg_powers_renew = []
agg_powers_no_renew = []
for i in range(len(agg_hourly_data)):
    time = i % 24
    # exclude heater and solar thermal ones
    agg_power_renew = max(agg_hourly_data[i] - boiler_hourly_data[i] - solar_hourly_data[i] - renewable_hourly_data[i], 0)
    excluded_powers_renew.append(agg_power_renew)

    agg_power = agg_hourly_data[i] - boiler_hourly_data[i] - solar_hourly_data[i]
    excluded_powers_no_renew.append(agg_power)

    if 4 <= time < 8:
        agg_power_renew += heater_power
        agg_power += heater_power
    agg_powers_renew.append(agg_power_renew)
    agg_powers_no_renew.append(agg_power)

np.save("data/excluded_power_renew_kWh_14_15.npy", excluded_powers_renew)
np.save("data/excluded_power_no_renew_kWh_14_15.npy", excluded_powers_no_renew)
np.save("data/new_agg_renew_kWh_14_15.npy", agg_powers_renew)
np.save("data/new_agg_no_renew_kWh_14_15.npy", agg_powers_no_renew)
