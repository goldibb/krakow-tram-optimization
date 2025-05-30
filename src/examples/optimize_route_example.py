import os
import sys
import logging
import geopandas as gpd
from typing import Tuple

# Dodanie ścieżki do katalogu src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.route_optimizer import RouteOptimizer, RouteConstraints
from visualization.route_visualizer import RouteVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_dir: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Wczytuje dane z plików GeoJSON.
    
    Args:
        data_dir: Katalog z danymi
        
    Returns:
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]: 
            DataFrames z budynkami, ulicami, przystankami i liniami
    """
    buildings_path = os.path.join(data_dir, 'buildings.geojson')
    streets_path = os.path.join(data_dir, 'streets.geojson')
    stops_path = os.path.join(data_dir, 'stops.geojson')
    lines_path = os.path.join(data_dir, 'lines.geojson')
    
    buildings_df = gpd.read_file(buildings_path)
    streets_df = gpd.read_file(streets_path)
    stops_df = gpd.read_file(stops_path)
    lines_df = gpd.read_file(lines_path)
    
    return buildings_df, streets_df, stops_df, lines_df

def main():
    # Wczytanie danych
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    buildings_df, streets_df, stops_df, lines_df = load_data(data_dir)
    
    # Definicja ograniczeń - bardziej elastyczne dla testowania
    constraints = RouteConstraints(
        min_distance_between_stops=200,  # zmniejszono z 300m 
        max_distance_between_stops=1200,  # zwiększono z 800m
        max_angle=60,  # zwiększono z 45°
        min_route_length=3,  # zmniejszono z 5
        max_route_length=25,  # zwiększono z 20
        min_total_length=1000,  # zmniejszono z 2000m
        max_total_length=15000,  # zwiększono z 10000m
        min_distance_from_buildings=3  # zmniejszono z 5m
    )
    
    # Inicjalizacja optymalizatora
    optimizer = RouteOptimizer(
        buildings_df=buildings_df,
        streets_df=streets_df,
        stops_df=stops_df,
        lines_df=lines_df,
        constraints=constraints,
        population_size=100,
        generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        population_weight=0.7,  # waga dla kryterium gęstości zaludnienia
        distance_weight=0.3     # waga dla kryterium odległości
    )
    
    # Wybór punktów startowego i końcowego
    start_point = (50.0647, 19.9450)  # Rondo Mogilskie
    end_point = (50.0720, 19.9570)    # Plac Centralny
    
    # Optymalizacja trasy
    logger.info("Rozpoczynam optymalizację trasy...")
    logger.info(f"Punkt startowy: {start_point}")
    logger.info(f"Punkt końcowy: {end_point}")
    logger.info(f"Liczba budynków: {len(buildings_df)}")
    logger.info(f"Liczba ulic: {len(streets_df)}")
    logger.info(f"Liczba przystanków: {len(stops_df)}")
    logger.info(f"Liczba linii: {len(lines_df)}")
    
    try:
        best_route, best_score = optimizer.optimize_route(
            start_point=start_point,
            end_point=end_point,
            num_stops=5,  # zmniejszono z 10 dla łatwiejszego testowania
            max_iterations=100  # zmniejszono z 1000 dla szybszego testowania
        )
        
        if best_route is None:
            logger.error("Nie znaleziono żadnej prawidłowej trasy!")
            return
            
        logger.info(f"Znaleziono najlepszą trasę z oceną: {best_score:.2f}")
        logger.info(f"Liczba przystanków w trasie: {len(best_route)}")
        
    except Exception as e:
        logger.error(f"Błąd podczas optymalizacji: {str(e)}")
        return
    
    # Wizualizacja wyników
    visualizer = RouteVisualizer(buildings_df, streets_df)
    
    # Tworzenie mapy gęstości
    density_map = visualizer.create_density_map(best_route)
    
    # Wizualizacja wyników
    m = visualizer.plot_optimization_results(
        best_route=best_route,
        density_map=density_map,
        bounds=visualizer.get_bounds(),
        score=best_score,
        save_path=os.path.join(data_dir, 'results', 'optimized_route.html')
    )
    
    logger.info("Wizualizacja została zapisana w katalogu results")

if __name__ == "__main__":
    main() 