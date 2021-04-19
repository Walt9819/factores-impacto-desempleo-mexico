# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 02:19:09 2021

@author: BALAMLAPTOP2
"""

from geopy.geocoders import Nominatim
import requests
import json
import pandas as pd

geolocator = Nominatim(user_agent='App de prueba')

class GooglePlaces(object):
    def __init__(self):
        super(GooglePlaces, self).__init__()
        self.apiKey = '----API KEY PERSONAL----'

    def search_places(self, types = None, location=None, near=None, radius = None, keyword = None):
        if near:
            geocod = geolocator.geocode(near)
            location = str(geocod.latitude)+','+str(geocod.longitude)
        endpoint_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            'radius': radius,
            'key': self.apiKey,
            'location': location,
            'type': types,
            'keyword': keyword
        }
        places = list()
        while True:
            try:
                res = requests.get(endpoint_url, params=params)
                res = json.loads(res.content)
                places.extend(res['results'])
                params['pagetoken'] = res['next_page_token']
            except KeyError:
                break
        return places
    
#if __name__ == '__main__':
        
Explorer = GooglePlaces()
df_gp = pd.DataFrame(columns=['icon','name','place_id','reference','scope','vicinity','rating','user_ratings_total', 'lat', 'lng'])
for nearp in ['Mexico City']:
    places = Explorer.search_places(near=nearp, radius=50000, types='restaurant', keyword = 'restaurante')
    for place in places:
        df_gp = df_gp.append({'icon': place['icon'],'name': place['name'],'place_id': place['place_id'],'reference': place['reference'],'scope': place['scope'],'vicinity': place['vicinity'],'rating': place['rating'],'user_ratings_total': place['user_ratings_total'], 'lat': place['geometry']['location']['lat'], 'lng': place['geometry']['location']['lng']}, ignore_index = True)

df_gp.to_csv(r'data/google_places.csv', index = False, encoding = 'utf-8') 