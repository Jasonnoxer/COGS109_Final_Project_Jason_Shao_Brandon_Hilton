import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Read in Data
TicketData = pd.read_csv('Data/MasterTicketData.csv')

# Initial Data frame characteristics
num_events = TicketData['event_id'].count()

# Remove rows w/o face_value
TicketData = TicketData[~TicketData['face_value'].str.contains('NoPriceClass')]
TicketData = TicketData[~TicketData['face_value'].str.contains('NoResults')]
TicketData = TicketData[~TicketData['face_value'].str.contains('NoRegexp')]
num_events_w_face_values = TicketData['event_id'].count()

print(f"Total number of events:{num_events}")
print(f"Total number of events with face values: {num_events_w_face_values}")

TicketData['face_value'] = TicketData['face_value'].map(lambda x: x.lstrip('$'))
TicketData['face_value'] = pd.to_numeric(TicketData['face_value'])

# Get rid of event_id #9478397 which is an outlier (stubhub ticket price is $711,771)
TicketData = TicketData[TicketData['event_id'] != 9478397]

print(f"Range of face_value: {TicketData['face_value'].min()} ~ {TicketData['face_value'].max()}")
print(f"Range of minimum price of a ticket on StubHub: {TicketData['min_price'].min()} ~ {TicketData['min_price'].max()}")
print(f"Range of maximum price of a ticket on StubHub: {TicketData['max_price'].min()} ~ {TicketData['max_price'].max()}")


# Get rid of rows where Echonest did not return any data
TicketData = TicketData[TicketData['num_news'] != 'error_5']
# Get rid of rows where num_years_active has a null value
TicketData = TicketData[TicketData['num_years_active'].isnull() == False]
# Get rid of rows where FV_delta_log is null (meaning FV_delta was negative)
TicketData = TicketData[TicketData['face_value'] < TicketData['min_price']]
print(TicketData.count())


# Uncomment this in order to save processed dataframe as CSV
TicketData.to_csv(path_or_buf="Data/Clean_Data.csv", index=False)

