import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.sourcing_data import TramData, OpenStreetMapData
import geopandas as gpd
from shapely.geometry import Point, LineString
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_data():
    """Zapisuje dane do plików GeoJSON."""
    logger.info("Pobieranie danych...")
    tram_data = TramData()
    osm_data = OpenStreetMapData()
    
    # Tworzenie katalogu data jeśli nie istnieje
    os.makedirs("data", exist_ok=True)
    
    # Zapisanie budynków
    logger.info("Zapisywanie budynków...")
    osm_data.buildings_df.to_file("data/buildings.geojson", driver="GeoJSON")
    
    # Zapisanie ulic
    logger.info("Zapisywanie ulic...")
    osm_data.streets_df.to_file("data/streets.geojson", driver="GeoJSON")
    
    # Konwersja i zapisanie przystanków
    logger.info("Zapisywanie przystanków...")
    stops_gdf = gpd.GeoDataFrame(
        tram_data.stops_df,
        geometry=[Point(lon, lat) for lat, lon in zip(tram_data.stops_df['latitude'], tram_data.stops_df['longitude'])],
        crs="EPSG:4326"
    )
    stops_gdf.to_file("data/stops.geojson", driver="GeoJSON")
    
    # Konwersja i zapisanie linii
    logger.info("Zapisywanie linii...")
    lines_data = []
    for line, coords in tram_data.mpk_sourcing.lines_stops_coordinates.items():
        if coords:  # sprawdzenie czy lista nie jest pusta
            line_geom = LineString([(lon, lat) for lat, lon in coords])
            lines_data.append({
                'line': line,
                'geometry': line_geom
            })
    lines_gdf = gpd.GeoDataFrame(lines_data, crs="EPSG:4326")
    lines_gdf.to_file("data/lines.geojson", driver="GeoJSON")
    
    logger.info("Wszystkie dane zostały zapisane w katalogu data/")

if __name__ == "__main__":
    save_data() 