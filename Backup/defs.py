import time
from geopy.geocoders import Nominatim
import geopy
import ssl
import certifi

ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx

geolocator = Nominatim(user_agent='geopiExercises', timeout=3)

def get_data(x):
    index, row = x
    time.sleep(0.5)
    response = geolocator.reverse(row['query'])
    address = response.raw['address']

    try:
        place_id = response.raw['place_id'] if 'place_id' in response.raw else 'NA'
        osm_type = response.raw['osm_type'] if 'osm_type' in response.raw else 'NA'
        house_number = address['house_number'] if 'house_number' in address else 'NA'
        road = address['road'] if 'road' in address else 'NA'
        neighbourhood = address['neighbourhood'] if 'neighbourhood' in address else 'NA'
        city = address['city'] if 'city' in address else 'NA'
        county = address['county'] if 'county' in address else 'NA'
        state = address['state'] if 'state' in address else 'NA'
        country = address['country'] if 'country' in address else 'NA'
        country_code = address['country_code'] if 'country_code' in address else 'NA'

        return place_id, osm_type, house_number, road, neighbourhood, city, county, state, country, country_code

    except:
        return None, None, None, None, None, None, None, None, None, None
