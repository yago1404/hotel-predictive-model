import pandas as pd
from sklearn.svm import SVC
import pickle as pk
from sklearn.model_selection import cross_val_score


date_month_normalized_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
                             'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}


def training_model():
    classifier = SVC()
    frame = pd.read_csv('hotel_bookings.csv')
    columns_names = ['adults', 'children', 'babies', 'lead_time', 'arrival_date_month']
    training_list = []

    print('Normalizing months...')
    index = 0
    for month in frame['arrival_date_month']:
        frame['arrival_date_month'][index] = date_month_normalized_map[month]
        index = index + 1
    print('Months Normalized')

    adults_mean = frame['adults'].mean()
    children_mean = frame['children'].mean()
    babies_mean = frame['babies'].mean()
    lead_time_mean = frame['lead_time'].mean()
    arrival_date_month_mean = frame['arrival_date_month'].mean()

    frame['adults'].fillna(value=adults_mean, inplace=True)
    frame['children'].fillna(value=children_mean, inplace=True)
    frame['babies'].fillna(value=babies_mean, inplace=True)
    frame['lead_time'].fillna(value=lead_time_mean, inplace=True)
    frame['arrival_date_month'].fillna(value=arrival_date_month_mean, inplace=True)

    print('Training model...')
    for current_line in range(len(frame['adults'][0: 95000])):
        current = []
        for column_name in columns_names:
            current.append(frame[column_name][current_line])
        training_list.append(current)
    classifier.fit(training_list, frame['is_canceled'][0: 95000])
    print('Training ended')

    print('Saving model...')
    pk.dump(classifier, open('model.clf', 'wb'))
    print('Saved')

    cross_score = cross_val_score(classifier, training_list, frame['is_canceled'][0: 95000])
    print(cross_score)
