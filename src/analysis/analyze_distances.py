import json
import pandas as pd
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

with open('data/stops.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

tram_stops = []
seen_coords = set()

for f in data['features']:
    if f['properties']['category'] == 'tram':
        coords = tuple(f['geometry']['coordinates'])
        if coords not in seen_coords:
            seen_coords.add(coords)
            tram_stops.append(f)

distances = []
for i, stop1 in enumerate(tram_stops):
    for j, stop2 in enumerate(tram_stops[i+1:], i+1):
        lat1 = stop1['geometry']['coordinates'][1]
        lon1 = stop1['geometry']['coordinates'][0]
        lat2 = stop2['geometry']['coordinates'][1]
        lon2 = stop2['geometry']['coordinates'][0]
        
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        if not pd.isna(distance):
            distances.append(distance)

df = pd.Series(distances)
print(f'Liczba przystanków tramwajowych: {len(tram_stops)}')
print(f'Statystyki odległości (w metrach):')
print(f'Min: {df.min():.0f}')
print(f'Q1: {df.quantile(0.25):.0f}')
print(f'Mediana: {df.median():.0f}')
print(f'Q3: {df.quantile(0.75):.0f}')
print(f'Max: {df.max():.0f}')
print(f'Średnia: {df.mean():.0f}')
print('')
print('Percentyle:')
for p in [1, 5, 10, 20, 30]:
    print(f'{p}%: {df.quantile(p/100):.0f}m') 