import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopandas as gpd
from src.optimization.route_optimizer import RouteOptimizer, RouteConstraints
from src.visualization.route_visualizer import RouteVisualizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_dir: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Wczytuje dane z plików GeoJSON.
    
    Args:
        data_dir: Katalog z danymi
        
    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]: 
            DataFrames z budynkami, ulicami, przystankami i liniami
    """
    buildings_path = os.path.join(data_dir, 'buildings.geojson')
    streets_path = os.path.join(data_dir, 'streets.geojson')
    stops_path = os.path.join(data_dir, 'stops.geojson')
    lines_path = os.path.join(data_dir, 'lines.geojson')
    
    logger.info("Wczytywanie danych z plików GeoJSON...")
    buildings_df = gpd.read_file(buildings_path)
    streets_df = gpd.read_file(streets_path)
    stops_df = gpd.read_file(stops_path)
    lines_df = gpd.read_file(lines_path)
    
    logger.info(f"Wczytano {len(buildings_df)} budynków")
    logger.info(f"Wczytano {len(streets_df)} ulic")
    logger.info(f"Wczytano {len(stops_df)} przystanków")
    logger.info(f"Wczytano {len(lines_df)} linii tramwajowych")
    
    return buildings_df, streets_df, stops_df, lines_df

def main():
    # Wczytanie danych
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    buildings_df, streets_df, stops_df, lines_df = load_data(data_dir)
    
    # Konfiguracja ograniczeń
    constraints = RouteConstraints(
        min_distance_between_stops=200,  # 200m między przystankami
        max_distance_between_stops=1500,  # 1500m między przystankami
        max_angle=60,  # maksymalny kąt zakrętu
        min_route_length=3,  # minimalna liczba przystanków
        max_route_length=20,  # maksymalna liczba przystanków
        min_total_length=1000,  # minimalna długość trasy
        max_total_length=15000,  # maksymalna długość trasy
        min_distance_from_buildings=3  # minimalna odległość od budynków
    )
    
    # Inicjalizacja optymalizatora
    logger.info("Inicjalizacja optymalizatora...")
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
    
    # Uruchomienie optymalizacji
    logger.info("Rozpoczynam optymalizację...")
    
    # Wybór punktów startowego i końcowego
    start_point = (50.0647, 19.9450)  # Rondo Mogilskie
    end_point = (50.0720, 19.9570)    # Plac Centralny
    
    best_route, best_score = optimizer.optimize_route(
        start_point=start_point,
        end_point=end_point,
        num_stops=5,  # liczba przystanków
        max_iterations=100  # zmniejszono dla szybszego testowania
    )
    
    if best_route is None:
        logger.error("Nie znaleziono poprawnej trasy!")
        return
        
    logger.info(f"Znaleziono trasę z oceną: {best_score:.2f}")
    
    # Inicjalizacja wizualizatora
    visualizer = RouteVisualizer(buildings_df, streets_df)
    
    # Obliczenie granic obszaru
    bounds = (
        min(lon for lat, lon in best_route),
        min(lat for lat, lon in best_route),
        max(lon for lat, lon in best_route),
        max(lat for lat, lon in best_route)
    )
    
    # Tworzenie mapy gęstości
    density_map = visualizer.create_density_map(best_route)
    
    # Wizualizacja wyników
    logger.info("Generowanie wizualizacji...")
    
    # Utworzenie katalogu results jeśli nie istnieje
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    m = visualizer.plot_optimization_results(
        best_route=best_route,
        density_map=density_map,
        bounds=bounds,
        score=best_score,
        save_path=os.path.join(results_dir, "optimized_route.html")
    )
    
    logger.info(f"Wizualizacja zapisana w {os.path.join(results_dir, 'optimized_route.html')}")

if __name__ == "__main__":
    main() 