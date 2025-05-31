import json
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    try:
        # Check for invalid coordinates
        if any(pd.isna([lat1, lon1, lat2, lon2])):
            return np.nan
            
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371  # Radius of earth in kilometers
        return c * r * 1000  # Convert to meters
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return np.nan

def analyze_stops():
    # Read the GeoJSON file
    with open('data/stops.geojson', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter only tram stops and remove duplicates
    tram_stops = []
    seen_coords = set()
    
    for f in data['features']:
        if f['properties']['category'] == 'tram':
            coords = tuple(f['geometry']['coordinates'])
            # Only add if we haven't seen these coordinates before
            if coords not in seen_coords:
                seen_coords.add(coords)
                tram_stops.append(f)
    
    print(f"\nTotal unique tram stops: {len(tram_stops)}")
    
    # Calculate distances between all pairs of stops
    distances = []
    for i, stop1 in enumerate(tram_stops):
        for j, stop2 in enumerate(tram_stops[i+1:], i+1):
            lat1 = stop1['geometry']['coordinates'][1]
            lon1 = stop1['geometry']['coordinates'][0]
            lat2 = stop2['geometry']['coordinates'][1]
            lon2 = stop2['geometry']['coordinates'][0]
            
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            if not pd.isna(distance):  # Only add valid distances
                distances.append({
                    'stop1': stop1['properties']['stop'],
                    'stop2': stop2['properties']['stop'],
                    'distance': distance
                })
    
    # Convert to DataFrame and sort by distance
    df = pd.DataFrame(distances)
    df = df.sort_values('distance')
    
    # Print the 10 closest pairs of stops
    print("\n10 closest pairs of tram stops:")
    print(df.head(10).to_string(index=False))
    
    # Print some statistics
    print("\nDistance statistics (in meters):")
    print(f"Minimum distance: {df['distance'].min():.2f}")
    print(f"Maximum distance: {df['distance'].max():.2f}")
    print(f"Average distance: {df['distance'].mean():.2f}")
    print(f"Median distance: {df['distance'].median():.2f}")

if __name__ == "__main__":
    analyze_stops() 