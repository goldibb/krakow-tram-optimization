#!/usr/bin/env python3
"""
SZYBKI test optymalizacji wielu tras tramwajowych.
Używa zoptymalizowanych parametrów dla praktycznego użycia.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import geopandas as gpd
from src.optimization.route_optimizer import RouteOptimizer, RouteConstraints
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

def test_fast_optimization():
    """Testuje szybką optymalizację wielu tras."""
    logger.info("=== 🚀 TEST SZYBKIEJ OPTYMALIZACJI TRAS ===")
    
    # Wczytaj dane
    buildings_df, streets_df, stops_df, lines_df = load_test_data()
    
    # Optymalne ograniczenia dla szybkości
    constraints = RouteConstraints(
        min_distance_between_stops=300,   # Większe odstępy = mniej przystanków
        max_distance_between_stops=1200,  
        max_angle=45,                     # Prostsze trasy
        min_route_length=3,               
        max_route_length=8,               # Krótsze trasy = szybciej
        min_total_length=1000,            
        max_total_length=6000,            # Krótsze trasy
        min_distance_from_buildings=5,    
        angle_weight=0.2
    )
    
    # Inicjalizacja z szybkimi parametrami
    logger.info("Inicjalizacja optymalizatora...")
    start_init = time.time()
    
    optimizer = RouteOptimizer(
        buildings_df=buildings_df,
        streets_df=streets_df,
        stops_df=stops_df,
        lines_df=lines_df,
        constraints=constraints,
        population_size=50,    # Domyślne - zostaną tymczasowo zmienione
        generations=30,        # Domyślne - zostaną tymczasowo zmienione  
        mutation_rate=0.2,     # Wyższa mutacja = szybsza eksploracja
        crossover_rate=0.8,
        population_weight=0.5,  # Zbalansowane wagi
        distance_weight=0.3,    
        angle_weight=0.2
    )
    
    init_time = time.time() - start_init
    logger.info(f"Inicjalizacja zajęła: {init_time:.2f} sekund")
    
    # SZYBKA optymalizacja wielu tras
    logger.info("\n--- SZYBKA OPTYMALIZACJA 3 TRAS ---")
    
    start_time = time.time()
    routes = optimizer.optimize_multiple_routes_fast(num_routes=3)
    total_time = time.time() - start_time
    
    logger.info(f"\n🏁 WYNIKI:")
    logger.info(f"   Łączny czas: {total_time:.1f} sekund")
    logger.info(f"   Średnio na trasę: {total_time/3:.1f} sekund") 
    logger.info(f"   Znalezionych tras: {len(routes)}")
    
    # Analiza wyników
    all_stops = set()
    total_route_length = 0
    
    for i, (route, score) in enumerate(routes):
        route_stops = optimizer._extract_stops_from_route(route)
        route_length = optimizer._calculate_total_length(route)
        total_route_length += route_length
        
        logger.info(f"\n📍 Trasa {i+1}:")
        logger.info(f"   Wynik: {score:.3f}")
        logger.info(f"   Punktów: {len(route)}")
        logger.info(f"   Przystanków: {len(route_stops)}")
        logger.info(f"   Długość: {route_length/1000:.2f} km")
        
        # Sprawdź szczegóły
        density_score = optimizer.calculate_density_score(route)
        distance_score = optimizer.calculate_distance_score(route)
        angle_score = optimizer.calculate_angle_score(route)
        
        logger.info(f"   Gęstość: {density_score:.3f}")
        logger.info(f"   Odległości: {distance_score:.3f}")
        logger.info(f"   Prostota: {angle_score:.3f}")
        
        # Dodaj przystanki do sprawdzenia unikatowości
        for stop in route_stops:
            normalized = (round(stop[0], 6), round(stop[1], 6))
            all_stops.add(normalized)
    
    # Sprawdź unikatowość
    expected_stops = sum(len(optimizer._extract_stops_from_route(route)) for route, _ in routes)
    unique_stops = len(all_stops)
    
    logger.info(f"\n🎯 SPRAWDZENIE UNIKATOWOŚCI:")
    logger.info(f"   Oczekiwanych przystanków: {expected_stops}")
    logger.info(f"   Unikalnych przystanków: {unique_stops}")
    logger.info(f"   Czy wszystkie unikalne: {'✅ TAK' if expected_stops == unique_stops else '❌ NIE'}")
    
    # Podsumowanie wydajności
    logger.info(f"\n📊 PODSUMOWANIE WYDAJNOŚCI:")
    logger.info(f"   Łączna długość tras: {total_route_length/1000:.2f} km")
    logger.info(f"   Czas na km trasy: {total_time/(total_route_length/1000):.1f} s/km")
    logger.info(f"   Używanych przystanków: {len(optimizer.used_stops)}")
    
    # Oszacowanie czasu dla standardowych parametrów
    standard_evals = 100 * 50 * 3  # population * generations * routes
    fast_evals = 20 * 15 * 3       # szybkie parametry (z uwzględnieniem early stopping ~10 gen)
    actual_speedup = standard_evals / fast_evals
    estimated_standard_time = total_time * actual_speedup
    
    logger.info(f"\n⚡ PORÓWNANIE Z STANDARDOWYMI PARAMETRAMI:")
    logger.info(f"   Szybkie ewaluacje: {fast_evals}")
    logger.info(f"   Standardowe ewaluacje: {standard_evals}")
    logger.info(f"   Przyspieszenie: {actual_speedup:.1f}x")
    logger.info(f"   Szacowany czas standardowy: {estimated_standard_time/60:.1f} minut")
    
    return routes, total_time

if __name__ == "__main__":
    try:
        routes, exec_time = test_fast_optimization()
        
        if len(routes) > 0:
            logger.info(f"\n🎉 Test zakończony sukcesem w {exec_time:.1f}s!")
            logger.info(f"✅ Znaleziono {len(routes)} optymalnych tras")
        else:
            logger.error("❌ Nie znaleziono żadnych tras")
            
    except Exception as e:
        logger.error(f"❌ Błąd podczas testu: {str(e)}")
        import traceback
        traceback.print_exc() 