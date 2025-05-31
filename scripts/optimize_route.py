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
    
    # KONFIGURACJA OGRANICZEŃ - ZAKTUALIZOWANE NA PODSTAWIE ANALIZY DANYCH KRAKOWA
    constraints = RouteConstraints(
        # REALISTYCZNE ODLEGŁOŚCI (analiza 21 linii: mediana 495m, percentile 25-75: 393-621m)
        min_distance_between_stops=350,   # Nieco luźniej niż 25th percentile (393m)
        max_distance_between_stops=700,   # Bardziej elastycznie niż 75th percentile (621m)
        
        # REALISTYCZNE DŁUGOŚCI TRAS (analiza: min 1.1km, max 24.4km, średnia 14.5km)
        min_total_length=1500,            # Sensowne minimum (1.5km)
        max_total_length=15000,           # Umiarkowane dla hackathonu (15km)
        
        # REALISTYCZNA LICZBA PRZYSTANKÓW (analiza: 4-37 przystanków, średnia 24)
        min_route_length=4,               # Minimum jak w realnych danych
        max_route_length=15,              # Umiarkowane dla hackathonu
        
        # ZACHOWANE ZAŁOŻENIA HACKATHONU + BEZPIECZEŃSTWO
        max_angle=45.0,                   # Proste trasy (wymaganie #3)
        min_distance_from_buildings=5.0   # ZWIĘKSZONE bezpieczeństwo (było 3)
    )
    
    # Inicjalizacja optymalizatora z REALISTYCZNYMI parametrami
    logger.info("Inicjalizacja optymalizatora z realistycznymi parametrami Krakowa...")
    optimizer = RouteOptimizer(
        buildings_df=buildings_df,
        streets_df=streets_df,
        stops_df=stops_df,
        lines_df=lines_df,
        constraints=constraints,
        population_size=50,               # Zmniejszone dla szybszego działania
        generations=20,                   # Zmniejszone dla szybszego działania
        mutation_rate=0.15,              # Nieco więcej mutacji
        crossover_rate=0.8,
        population_weight=0.6,            # Nieco mniej wagi na gęstość
        distance_weight=0.3,              # Więcej na odległości
        angle_weight=0.1                  # Waga dla prostoty tras
    )
    
    # Uruchomienie optymalizacji
    logger.info("🚊 Rozpoczynam optymalizację z mechanizmami bezpieczeństwa...")
    logger.info("🔒 Automatyczne sprawdzanie kolizji z budynkami jest włączone")
    
    # Wybór punktów startowego i końcowego z najgęściej zaludnionych przystanków
    # Użyjemy TOP przystanków znalezionych przez optymalizator
    top_density_stops = optimizer._find_top_density_stops(top_n=5)
    
    start_point = top_density_stops[0]  # Najgęściej zaludniony przystanek
    end_point = top_density_stops[4]    # Piąty najgęściej zaludniony przystanek
    
    logger.info(f"Punkt startowy: {start_point}")
    logger.info(f"Punkt końcowy: {end_point}")
    logger.info(f"Liczba budynków: {len(buildings_df)}")
    logger.info(f"Liczba ulic: {len(streets_df)}")
    logger.info(f"Liczba przystanków: {len(stops_df)}")
    logger.info(f"Liczba linii: {len(lines_df)}")
    
    best_route, best_score = optimizer.optimize_route(
        start_point=start_point,
        end_point=end_point,
        num_stops=8,  # REALISTYCZNA liczba przystanków dla jednej trasy
        max_iterations=500  # Zmniejszone dla szybszego działania
    )
    
    if best_route is None:
        logger.error("Nie znaleziono poprawnej trasy!")
        return
        
    logger.info(f"Znaleziono trasę z oceną: {best_score:.2f}")
    
    # 🔒 SPRAWDZENIE BEZPIECZEŃSTWA ZNALEZIONEJ TRASY
    logger.info("🔍 Sprawdzanie bezpieczeństwa znalezionej trasy...")
    is_safe, safety_msg = optimizer._validate_route_safety(best_route)
    
    if is_safe:
        logger.info(f"✅ TRASA BEZPIECZNA: {safety_msg}")
    else:
        logger.warning(f"⚠️ UWAGA - {safety_msg}")
        
    # Dodatkowe sprawdzenie kolizji z budynkami
    has_collision = optimizer._check_collision_with_buildings(best_route)
    if has_collision:
        logger.warning("⚠️ WYKRYTO KOLIZJĘ Z BUDYNKAMI!")
    else:
        logger.info("✅ Brak kolizji z budynkami")
    
    # Szczegółowe informacje o trasie
    total_length = optimizer._calculate_total_length(best_route)
    logger.info(f"📏 Długość trasy: {total_length:.0f} metrów")
    logger.info(f"🚏 Liczba przystanków: {len(best_route)}")
    
    # Inicjalizacja wizualizatora
    visualizer = RouteVisualizer(buildings_df, streets_df)
    
    # Obliczenie granic obszaru
    bounds = (
        min(lon for lat, lon in best_route),
        min(lat for lat, lon in best_route),
        max(lon for lat, lon in best_route),
        max(lat for lat, lon in best_route)
    )
    
    # Wizualizacja wyników
    logger.info("Generowanie wizualizacji...")
    
    # Utworzenie katalogu results jeśli nie istnieje
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Tworzenie mapy
    m = visualizer.create_base_map()
    visualizer.plot_route(best_route, m, route_name="Zoptymalizowana trasa", color='red')
    
    # Zapisanie mapy
    map_path = os.path.join(results_dir, "optimized_route.html")
    m.save(map_path)
    
    logger.info(f"Wizualizacja zapisana w {map_path}")

if __name__ == "__main__":
    main() 