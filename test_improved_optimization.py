#!/usr/bin/env python3
"""
Skrypt testowy dla ulepszonego systemu optymalizacji tras tramwajowych.
Sprawdza wszystkie nowe funkcje: unikatowość przystanków, połączenie tras, minimalizację kątów.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import geopandas as gpd
from src.optimization.route_optimizer import RouteOptimizer, RouteConstraints
from src.visualization.route_visualizer import RouteVisualizer
import logging
import time

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data():
    """Wczytuje dane testowe."""
    logger.info("Wczytywanie danych testowych...")
    
    data_dir = 'data'
    buildings_df = gpd.read_file(os.path.join(data_dir, 'buildings.geojson'))
    streets_df = gpd.read_file(os.path.join(data_dir, 'streets.geojson'))
    stops_df = gpd.read_file(os.path.join(data_dir, 'stops.geojson'))
    lines_df = gpd.read_file(os.path.join(data_dir, 'lines.geojson'))
    
    logger.info(f"Wczytano: {len(buildings_df)} budynków, {len(streets_df)} ulic, "
               f"{len(stops_df)} przystanków, {len(lines_df)} linii")
    
    return buildings_df, streets_df, stops_df, lines_df

def test_route_optimizer():
    """Testuje ulepszony optymalizator tras."""
    logger.info("=== TEST ULEPSZONEGO OPTYMALIZATORA TRAS ===")
    
    # Wczytaj dane
    buildings_df, streets_df, stops_df, lines_df = load_test_data()
    
    # Konfiguracja ograniczeń zgodnie z wymaganiami
    constraints = RouteConstraints(
        min_distance_between_stops=200,  # 200m między przystankami
        max_distance_between_stops=1500,  # 1500m między przystankami
        max_angle=60,  # maksymalny kąt zakrętu
        min_route_length=3,  # minimalna liczba przystanków
        max_route_length=15,  # maksymalna liczba przystanków (zmniejszone dla testów)
        min_total_length=1000,  # minimalna długość trasy
        max_total_length=10000,  # maksymalna długość trasy (zmniejszone dla testów)
        min_distance_from_buildings=3,  # minimalna odległość od budynków
        angle_weight=0.1
    )
    
    # Inicjalizacja optymalizatora
    logger.info("Inicjalizacja optymalizatora...")
    optimizer = RouteOptimizer(
        buildings_df=buildings_df,
        streets_df=streets_df,
        stops_df=stops_df,
        lines_df=lines_df,
        constraints=constraints,
        population_size=20,  # Małe dla szybkiego testu
        generations=10,      # Małe dla szybkiego testu
        mutation_rate=0.15,
        crossover_rate=0.8,
        population_weight=0.6,  # Gęstość zabudowy
        distance_weight=0.3,    # Odległość między przystankami  
        angle_weight=0.1        # Minimalizacja kątów
    )
    
    logger.info(f"Wagi kryteriów: gęstość={optimizer.population_weight:.2f}, "
               f"odległość={optimizer.distance_weight:.2f}, kąty={optimizer.angle_weight:.2f}")
    
    # Test 1: Optymalizacja pojedynczej trasy
    logger.info("\n--- TEST 1: Optymalizacja pojedynczej trasy ---")
    start_time = time.time()
    
    best_route, best_score = optimizer.optimize()
    
    optimization_time = time.time() - start_time
    logger.info(f"Czas optymalizacji: {optimization_time:.2f} sekund")
    
    if best_route:
        logger.info(f"✅ Znaleziono trasę z wynikiem: {best_score:.3f}")
        logger.info(f"Liczba punktów w trasie: {len(best_route)}")
        
        # Wyodrębnij główne przystanki
        route_stops = optimizer._extract_stops_from_route(best_route)
        logger.info(f"Liczba głównych przystanków: {len(route_stops)}")
        
        # Sprawdź szczegółowe wyniki
        density_score = optimizer.calculate_density_score(best_route)
        distance_score = optimizer.calculate_distance_score(best_route)
        angle_score = optimizer.calculate_angle_score(best_route)
        total_length = optimizer._calculate_total_length(best_route)
        
        logger.info(f"Szczegóły: gęstość={density_score:.3f}, odległość={distance_score:.3f}, "
                   f"kąty={angle_score:.3f}, długość={total_length/1000:.2f}km")
    else:
        logger.error("❌ Nie udało się znaleźć trasy")
    
    # Test 2: Optymalizacja wielu tras z unikatowością
    logger.info("\n--- TEST 2: Optymalizacja wielu tras z unikatowością ---")
    
    # Resetuj używane przystanki
    optimizer.reset_used_stops()
    
    start_time = time.time()
    multiple_routes = optimizer.optimize_multiple_routes(num_routes=2)  # Test z 2 trasami
    multi_optimization_time = time.time() - start_time
    
    logger.info(f"Czas optymalizacji wielu tras: {multi_optimization_time:.2f} sekund")
    logger.info(f"Znaleziono {len(multiple_routes)} tras")
    
    # Sprawdź unikatowość przystanków
    all_used_stops = set()
    duplicates_found = False
    
    for i, (route, score) in enumerate(multiple_routes):
        logger.info(f"\nTrasa {i+1}: wynik={score:.3f}, punktów={len(route)}")
        
        route_stops = optimizer._extract_stops_from_route(route)
        logger.info(f"  Główne przystanki: {len(route_stops)}")
        
        # Sprawdź unikatowość
        for stop in route_stops:
            normalized = (round(stop[0], 6), round(stop[1], 6))
            if normalized in all_used_stops:
                logger.error(f"  ❌ Znaleziono duplikat przystanku: {stop}")
                duplicates_found = True
            else:
                all_used_stops.add(normalized)
    
    if not duplicates_found:
        logger.info("✅ Wszystkie przystanki są unikatowe!")
    
    logger.info(f"Łącznie użytych przystanków: {len(all_used_stops)}")
    logger.info(f"Przystanki w systemie: {len(optimizer.used_stops)}")
    
    # Test 3: Test funkcji pomocniczych
    logger.info("\n--- TEST 3: Funkcje pomocnicze ---")
    
    if best_route and len(best_route) >= 2:
        # Test połączenia tras
        test_stops = optimizer._extract_stops_from_route(best_route)[:3]  # Weź pierwsze 3 przystanki
        
        logger.info(f"Test połączenia dla {len(test_stops)} przystanków...")
        connected_route = optimizer._create_connected_route(test_stops)
        logger.info(f"Połączona trasa ma {len(connected_route)} punktów")
        
        # Test zapewnienia unikatowości
        logger.info("Test zapewnienia unikatowości...")
        test_route_with_duplicates = test_stops + test_stops[:2]  # Dodaj duplikaty
        unique_route = optimizer._ensure_unique_stops(test_route_with_duplicates)
        logger.info(f"Przed: {len(test_route_with_duplicates)} punktów, "
                   f"Po: {len(unique_route)} punktów")
        
        if len(unique_route) < len(test_route_with_duplicates):
            logger.info("✅ Duplikaty zostały usunięte")
        
        # Test obliczania kąta
        if len(best_route) >= 3:
            angle_score = optimizer.calculate_angle_score(best_route)
            logger.info(f"Wynik prostoty trasy: {angle_score:.3f}")
    
    # Podsumowanie
    logger.info("\n=== PODSUMOWANIE TESTÓW ===")
    logger.info(f"✅ Test optymalizacji pojedynczej trasy: {'PASSED' if best_route else 'FAILED'}")
    logger.info(f"✅ Test optymalizacji wielu tras: {'PASSED' if multiple_routes else 'FAILED'}")
    logger.info(f"✅ Test unikatowości przystanków: {'PASSED' if not duplicates_found else 'FAILED'}")
    logger.info(f"📊 Łączny czas testów: {optimization_time + multi_optimization_time:.2f} sekund")

if __name__ == "__main__":
    try:
        test_route_optimizer()
        logger.info("\n🎉 Wszystkie testy zakończone!")
    except Exception as e:
        logger.error(f"❌ Błąd podczas testów: {str(e)}")
        import traceback
        traceback.print_exc() 