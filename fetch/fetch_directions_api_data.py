import numpy as np
import pandas as pd
import googlemaps
import json

from datetime import datetime


# Choose starting and ending times
start_time = "2020-09-01T04:00:00.464Z"
end_time = "2020-09-03T22:00:00.464Z"

YOUR_API_KEY = ""

starttime =  datetime.strptime(start_time,'%Y-%m-%dT%H:%M:%S.%fZ')
endtime =  datetime.strptime(end_time,'%Y-%m-%dT%H:%M:%S.%fZ')



# gmaps = googlemaps.Client(key=YOUR_API_KEY)
#
# # Look up an address with reverse geocoding
# origin = gmaps.reverse_geocode((51.4913,-0.08168))
# destination = gmaps.reverse_geocode((51.490469,-0.080686))
#
#
# # Request directions
# directions_result = gmaps.directions("Sydney Town Hall",
#                                      "Parramatta, NSW",
#                                      mode="driving",
#                                      units="metric",
#                                      traffic_model="best_guess",
#                                      departure_time=starttime)
