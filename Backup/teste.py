import pandas as pd
data = pd.read_csv('kc_house_data.csv')

import geopy
from geopy.geocoders import Nominatim

# Inicializar Nominatim API


import certifi
import ssl

ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx
geolocator = Nominatim(user_agent='geoapiExercise')
response = geolocator.reverse('47.5112,-122.257')
print(response)
# import time
# from multiprocessing import Pool
#
# data['query'] = data[['lat', 'long']].apply(lambda x: str(x['lat']) + ',' + str(x['long']), axis=1)
#
# import defs1
#
# df1 = data[['id', 'query']]
#
# p = Pool(3)
#
# start = time.process_time()
# df1[['place_id', 'ost_type', 'country', 'country_code']] = p.map(defs1.get_data, df1.iterrows())
# end = time.process_time()
#
# print(f'Time elapsed: {end - start}')
#
