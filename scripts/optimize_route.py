import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopandas as gpd
from src.optimization.route_optimizer import RouteOptimizer, RouteConstraints
from src.visualization.route_visualizer import RouteVisualizer
from scripts.sourcing_data import TramData, OpenStreetMapData
from shapely.geometry import Point, LineString
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Pobieranie danych
    logger.info("Pobieranie danych z MPK i OpenStreetMap...")
    tram_data = TramData()
    osm_data = OpenStreetMapData()
    
    # Konwersja danych do GeoDataFrame
    logger.info("Przygotowywanie danych do optymalizacji...")
    
    # Konwersja przystanków
    stops_gdf = gpd.GeoDataFrame(
        tram_data.stops_df,
        geometry=[Point(lon, lat) for lat, lon in zip(tram_data.stops_df['latitude'], tram_data.stops_df['longitude'])],
        crs="EPSG:4326"
    )
    
    # Konwersja linii
    lines_data = []
    for line, coords in tram_data.mpk_sourcing.lines_stops_coordinates.items():
        if coords:  # sprawdzenie czy lista nie jest pusta
            line_geom = LineString([(lon, lat) for lat, lon in coords])
            lines_data.append({
                'line': line,
                'geometry': line_geom
            })
    lines_gdf = gpd.GeoDataFrame(lines_data, crs="EPSG:4326")
    
    # Konfiguracja ograniczeń
    constraints = RouteConstraints(
        min_distance_between_stops=300,  # 300m między przystankami
        max_distance_between_stops=1000,  # 1000m między przystankami
        max_angle=45,  # maksymalny kąt zakrętu
        min_route_length=5,  # minimalna liczba przystanków
        max_route_length=15,  # maksymalna liczba przystanków
        min_total_length=2000,  # minimalna długość trasy
        max_total_length=10000,  # maksymalna długość trasy
        min_distance_from_buildings=5  # minimalna odległość od budynków
    )
    
    # Inicjalizacja optymalizatora
    logger.info("Inicjalizacja optymalizatora...")
    optimizer = RouteOptimizer(
        buildings_df=osm_data.buildings_df,
        streets_df=osm_data.streets_df,
        stops_df=stops_gdf,
        lines_df=lines_gdf,
        constraints=constraints,
        population_size=100,
        generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    # Uruchomienie optymalizacji
    logger.info("Rozpoczynam optymalizację...")
    best_route, best_score = optimizer.optimize()
    
    if best_route is None:
        logger.error("Nie znaleziono poprawnej trasy!")
        return
        
    logger.info(f"Znaleziono trasę z oceną: {best_score:.2f}")
    
    # Inicjalizacja wizualizatora
    visualizer = RouteVisualizer(osm_data.buildings_df, osm_data.streets_df)
    
    # Obliczenie granic obszaru
    bounds = (
        min(lon for lat, lon in best_route),
        min(lat for lat, lon in best_route),
        max(lon for lat, lon in best_route),
        max(lat for lat, lon in best_route)
    )
    
    # Generowanie mapy gęstości
    density_map = optimizer.density_calculator.get_density_map(
        grid_size=0.001,  # rozmiar siatki w stopniach
        bounds=bounds
    )
    
    # Wizualizacja wyników
    logger.info("Generowanie wizualizacji...")
    m = visualizer.plot_optimization_results(
        best_route=best_route,
        density_map=density_map,
        bounds=bounds,
        score=best_score,
        save_path="results/optimized_route.html"
    )
    
    logger.info("Wizualizacja zapisana w results/optimized_route.html")

if __name__ == "__main__":
    main() 