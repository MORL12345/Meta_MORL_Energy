import pandas as pd
from datetime import datetime
import numpy as np

# 读取数据文件
# 假设你的数据文件名为 'data.dat'，并且数据以空格分隔
file_path = '../dataset/house_1/channel_3.dat'
df = pd.read_csv(file_path, sep=" ", header=None)
df.columns = ['Timestamp', 'Power']

# 将时间戳转换为日期
# 假设时间戳是Unix时间戳（以秒为单位）
df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
result = df.query('Date == "2014-01-01 00:00:04"')  # 5123427 5,046,149
print(f"result:{result}")
result = df.query('Date == "2015-01-01 00:00:04"')  # 10010754
print(f"result:{result}")
print(df.iloc[0:2]["Timestamp"])
print(df.iloc[1]["Timestamp"] - df.iloc[0]["Timestamp"])

aggregated_hourly_powers = []
cnt = 1
hours = 0
day = 0
aggregated_hourly_power = 0
for index, row in df.iloc[5123305:10010632].iterrows():
    seconds = df.iloc[index + 1]["Timestamp"] - df.iloc[index]["Timestamp"]
    for _ in range(seconds):
        aggregated_hourly_power += row['Power']
        cnt += 1

        if cnt % 3600 == 0:
            print(f"day:{day}-hour:{hours % 24}\tpower:{aggregated_hourly_power / 3600}")
            hours += 1
            aggregated_hourly_powers.append(aggregated_hourly_power / (3600 * 1000))
            aggregated_hourly_power = 0
            if hours % 24 == 0:
                day += 1

print(f"max{max(aggregated_hourly_powers)}")
np.save("../simulator/data/solar_pump_hourly_power_kWh_14_15.npy", aggregated_hourly_powers)
for power in aggregated_hourly_powers:
    print(f"power:{power}")
print(df)
