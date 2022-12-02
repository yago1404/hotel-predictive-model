import pandas as pd
import matplotlib.pyplot as plt

from training_model import training_model

# plt.rcdefaults()
# import numpy as np
# import matplotlib.pyplot as plt
#
# df = pd.read_csv('hotel_bookings.csv')
#
# monthMap = {}
# index = 0
#
# for frame in df['arrival_date_month']:
#     if frame in monthMap:
#         monthMap[frame] = monthMap[frame] + 1
#         if df['is_canceled'][index] == 1:
#             monthMap[frame + 'Canceled'] = monthMap[frame + 'Canceled'] + 1
#     else:
#         monthMap[frame] = 1
#         if df['is_canceled'][index] == 1:
#             monthMap[frame + 'Canceled'] = 1
#         else:
#             monthMap[frame + 'Canceled'] = 0
#     index = index + 1
#
# print(monthMap)
#
# y_pos = np.arange(len(monthMap))
# performance = monthMap.values()
#
# plt.bar(y_pos, performance, align='center', color=['blue', 'red'])
# plt.xticks(y_pos, ['7', '', '8', '', '9', '', '10', '', '11', '', '12', '', '1', '', '2', '', '3', '', '4', '', '5', '', '6', ''])
# plt.title('For month')
#
# plt.show()
#
# print(df)

training_model()
