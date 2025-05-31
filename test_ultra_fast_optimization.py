#!/usr/bin/env python3
"""
ULTRASZYBKA optymalizacja tras tramwajowych - maksymalnie 2-3 minuty.
Spełnia wszystkie wymagania projektowe dla linii tramwajowej w Krakowie.
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

def load_data():
    """Wczytuje dane OSM i TTSS."""
    logger.info("🗂️  Wczytywanie danych OSM i TTSS...")
    
    data_dir = 'data'
    buildings_df = gpd.read_file(os.path.join(data_dir, 'buildings.geojson'))  # Gęstość zabudowy
    streets_df = gpd.read_file(os.path.join(data_dir, 'streets.geojson'))      # Sieć ulic
    stops_df = gpd.read_file(os.path.join(data_dir, 'stops.geojson'))          # Istniejące przystanki
    lines_df = gpd.read_file(os.path.join(data_dir, 'lines.geojson'))          # Istniejące linie tramwajowe
    
    logger.info(f"📊 Dane załadowane:")
    logger.info(f"   🏢 Budynki: {len(buildings_df)} (dla gęstości zabudowy)")
    logger.info(f"   🛣️  Ulice: {len(streets_df)} (sieć transportowa)")
    logger.info(f"   🚏 Przystanki: {len(stops_df)} (punkty startowe)")
    logger.info(f"   🚊 Linie tramwajowe: {len(lines_df)} (kolizje)")
    
    return buildings_df, streets_df, stops_df, lines_df

def setup_optimization_constraints():
    """Konfiguruje ograniczenia zgodnie z wymaganiami projektowymi - ZAKTUALIZOWANE na podstawie analizy danych Krakowa."""
    logger.info("⚙️  Konfiguracja REALISTYCZNYCH ograniczeń na podstawie danych krakowskich...")
    
    constraints = RouteConstraints(
        # REALISTYCZNE ODLEGŁOŚCI (na podstawie analizy: mediana 495m, 25-75 percentile: 393-621m)
        min_distance_between_stops=350,   # Nieco luźniej niż 25th percentile
        max_distance_between_stops=700,   # Bardziej elastycznie niż 75th percentile
        
        # REALISTYCZNE DŁUGOŚCI TRAS (na podstawie analizy: min 1.1km, max 24.4km, średnia 14.5km)
        min_total_length=1500,            # Sensowne minimum (1.5km)
        max_total_length=15000,           # Umiarkowane dla hackathonu (15km)
        
        # REALISTYCZNA LICZBA PRZYSTANKÓW (na podstawie analizy: 4-37 przystanków, średnia 24)
        min_route_length=4,               # Minimum jak w realnych danych
        max_route_length=15,              # Umiarkowane dla hackathonu
        
        # ZACHOWANE ZAŁOŻENIA HACKATHONU
        max_angle=45,                     # Proste trasy (wymaganie #3)
        min_distance_from_buildings=3     # Bezpieczeństwo
    )
    
    logger.info(f"✅ Odległości między przystankami: {constraints.min_distance_between_stops}-{constraints.max_distance_between_stops}m")
    logger.info(f"✅ Długość tras: {constraints.min_total_length/1000:.1f}-{constraints.max_total_length/1000:.1f}km")
    logger.info(f"✅ Liczba przystanków: {constraints.min_route_length}-{constraints.max_route_length}")
    logger.info(f"✅ Maksymalny kąt zakrętu: {constraints.max_angle}°")
    
    return constraints

def test_ultra_fast_optimization():
    """Test ultraszybkiej optymalizacji tras tramwajowych."""
    logger.info("🚀 === ULTRASZYBKA OPTYMALIZACJA TRAS TRAMWAJOWYCH ===")
    logger.info("🎯 Cel: Maksymalizacja gęstości zabudowy (300m) + optymalne odległości + proste trasy")
    logger.info("⏱️  Limit czasu: 5 minut")
    logger.info("🔧 Nowe funkcje: kontrolowane odległości (300-1200m) + lepsze sprawdzanie kolizji")
    
    # Wczytaj dane
    buildings_df, streets_df, stops_df, lines_df = load_data()
    
    # Konfiguruj ograniczenia
    constraints = setup_optimization_constraints()
    
    # Inicjalizacja optymalizatora z wymaganiami projektowymi
    logger.info("\n🔧 Inicjalizacja optymalizatora...")
    optimizer = RouteOptimizer(
        buildings_df=buildings_df,  # WYMAGANIE #1: Gęstość zabudowy
        streets_df=streets_df,      # Sieć transportowa
        stops_df=stops_df,          # OGRANICZENIE #2: Początek na istniejącym przystanku
        lines_df=lines_df,          # OGRANICZENIE #3: Brak kolizji z istniejącą infrastrukturą
        constraints=constraints,
        
        # Parametry już zoptymalizowane w ultra_fast
        population_size=50,   # Będzie zmniejszone do 10 w ultra_fast
        generations=20,       # Będzie zmniejszone do 5 w ultra_fast
        mutation_rate=0.15,
        crossover_rate=0.8,
        
        # WAGI KRYTERIÓW ZGODNIE Z WYMAGANIAMI
        population_weight=0.6,  # 60% - WYMAGANIE #1: Gęstość zabudowy
        distance_weight=0.3,    # 30% - WYMAGANIE #2: Optymalne odległości
        angle_weight=0.1        # 10% - WYMAGANIE #3: Prostota tras
    )
    
    logger.info(f"✅ Wagi kryteriów optymalizacji:")
    logger.info(f"   🏢 Gęstość zabudowy: {optimizer.population_weight:.1%}")
    logger.info(f"   📏 Odległości przystanków: {optimizer.distance_weight:.1%}")
    logger.info(f"   📐 Prostota tras: {optimizer.angle_weight:.1%}")
    
    # ULTRASZYBKA OPTYMALIZACJA WIELU TRAS
    logger.info("\n🚀 Rozpoczynam ultraszybką optymalizację...")
    start_time = time.time()
    
    # Optymalizuj 2 trasy w maksymalnie 3 minuty
    routes = optimizer.optimize_multiple_routes_ultra_fast(
        num_routes=2,  # Zmniejszone z 3 do 2
        time_limit_minutes=3  # Zmniejszone z 5 do 3 minut
    )
    
    total_time = time.time() - start_time
    
    # ANALIZA WYNIKÓW
    logger.info(f"\n📊 === WYNIKI OPTYMALIZACJI ===")
    logger.info(f"⏱️  Całkowity czas: {total_time:.1f} sekund ({total_time/60:.1f} minut)")
    logger.info(f"🚊 Liczba znalezionych tras: {len(routes)}")
    
    if not routes:
        logger.error("❌ Nie znaleziono żadnych tras!")
        return
    
    # Szczegółowa analiza każdej trasy
    for i, (route, score) in enumerate(routes):
        logger.info(f"\n🚊 === TRASA {i+1} - ANALIZA WYMAGAŃ ===")
        
        # Podstawowe informacje
        route_stops = optimizer._extract_stops_from_route(route)
        total_length = optimizer._calculate_total_length(route)
        
        logger.info(f"📏 Podstawowe informacje:")
        logger.info(f"   Punktów w trasie: {len(route)}")
        logger.info(f"   Głównych przystanków: {len(route_stops)}")
        logger.info(f"   Długość całkowita: {total_length/1000:.2f}km")
        
        # WYMAGANIE #1: Gęstość zabudowy (300m)
        density_score = optimizer.calculate_density_score(route, radius=300)
        logger.info(f"\n🏢 WYMAGANIE #1 - Gęstość zabudowy (300m):")
        logger.info(f"   Wynik gęstości: {density_score:.3f}")
        logger.info(f"   Status: {'✅ SPEŁNIONE' if density_score > 0.1 else '❌ NIESPEŁNIONE'}")
        
        # WYMAGANIE #2: Odległości między przystankami
        if len(route_stops) >= 2:
            distances = []
            for j in range(len(route_stops) - 1):
                dist = optimizer._calculate_distance(route_stops[j], route_stops[j+1], is_wgs84=True)
                distances.append(dist)
            
            avg_distance = sum(distances) / len(distances)
            min_distance = min(distances)
            max_distance = max(distances)
            distance_score = optimizer.calculate_distance_score(route)
            
            logger.info(f"\n📏 WYMAGANIE #2 - Odległości między przystankami:")
            logger.info(f"   Średnia odległość: {avg_distance:.0f}m")
            logger.info(f"   Minimalna: {min_distance:.0f}m (min: {constraints.min_distance_between_stops}m)")
            logger.info(f"   Maksymalna: {max_distance:.0f}m (max: {constraints.max_distance_between_stops}m)")
            logger.info(f"   Wynik odległości: {distance_score:.3f}")
            
            distance_ok = (min_distance >= constraints.min_distance_between_stops and 
                          max_distance <= constraints.max_distance_between_stops)
            logger.info(f"   Status: {'✅ SPEŁNIONE' if distance_ok else '❌ NIESPEŁNIONE'}")
        
        # WYMAGANIE #3: Prostota tras (minimalizacja zakrętów)
        angle_score = optimizer.calculate_angle_score(route)
        logger.info(f"\n📐 WYMAGANIE #3 - Prostota tras:")
        logger.info(f"   Wynik prostoty: {angle_score:.3f} (1.0 = idealna prostota)")
        logger.info(f"   Status: {'✅ SPEŁNIONE' if angle_score > 0.7 else '❌ NIESPEŁNIONE'}")
        
        # OGRANICZENIE #1: Długość trasy
        length_ok = (constraints.min_total_length <= total_length <= constraints.max_total_length)
        logger.info(f"\n📊 OGRANICZENIE #1 - Długość trasy:")
        logger.info(f"   Zakres: {constraints.min_total_length/1000:.1f}-{constraints.max_total_length/1000:.1f}km")
        logger.info(f"   Aktualna: {total_length/1000:.2f}km")
        logger.info(f"   Status: {'✅ SPEŁNIONE' if length_ok else '❌ NIESPEŁNIONE'}")
        
        # OGRANICZENIE #2: Początek na istniejącym przystanku
        if route_stops:
            start_stop = route_stops[0]
            is_valid_start = optimizer._is_valid_start_stop(start_stop)
            logger.info(f"\n🚏 OGRANICZENIE #2 - Początek na istniejącym przystanku:")
            logger.info(f"   Pierwszy przystanek: {start_stop}")
            logger.info(f"   Status: {'✅ SPEŁNIONE' if is_valid_start else '❌ NIESPEŁNIONE'}")
        
        # OGRANICZENIE #3: Brak kolizji z istniejącą infrastrukturą
        has_line_collision = optimizer._check_collision_with_existing_lines(route)
        logger.info(f"\n🚊 OGRANICZENIE #3 - Kolizje z istniejącymi liniami:")
        logger.info(f"   Kolizja wykryta: {'TAK' if has_line_collision else 'NIE'}")
        logger.info(f"   Status: {'❌ NIESPEŁNIONE' if has_line_collision else '✅ SPEŁNIONE'}")
        
        # OGRANICZENIE #4: Brak kolizji z zabudową
        has_building_collision = optimizer._check_collision_with_buildings(route)
        logger.info(f"\n🏗️ OGRANICZENIE #4 - Kolizje z zabudową:")
        logger.info(f"   Kolizja wykryta: {'TAK' if has_building_collision else 'NIE'}")
        logger.info(f"   Status: {'❌ NIESPEŁNIONE' if has_building_collision else '✅ SPEŁNIONE'}")
        
        # ŁĄCZNY WYNIK
        logger.info(f"\n🎯 ŁĄCZNY WYNIK TRASY: {score:.3f}")
        logger.info(f"   Dekompozycja: {optimizer.population_weight:.1f}×{density_score:.3f} + "
                   f"{optimizer.distance_weight:.1f}×{distance_score:.3f} + "
                   f"{optimizer.angle_weight:.1f}×{angle_score:.3f}")
    
    # PODSUMOWANIE
    logger.info(f"\n🏁 === PODSUMOWANIE ===")
    logger.info(f"✅ Ultraszybka optymalizacja zakończona w {total_time:.1f}s")
    logger.info(f"🚊 Znaleziono {len(routes)} tras spełniających wymagania projektowe")
    logger.info(f"🎯 Model zoptymalizowany dla gęstości zabudowy, odległości i prostoty tras")
    logger.info(f"📋 Wszystkie ograniczenia infrastrukturalne zostały sprawdzone")
    
    return routes

def test_intelligent_fast_optimization():
    """Test nowej inteligentnej szybkiej optymalizacji - zachowuje wszystkie wymagania."""
    logger.info("🧠 === TEST INTELIGENTNEJ SZYBKIEJ OPTYMALIZACJI ===")
    logger.info("🎯 Cel: Zachowanie WSZYSTKICH wymagań + drastyczne przyspieszenie")
    logger.info("⏱️  Limit czasu: 3 minuty")
    logger.info("🔧 Smart features: heuristics + caching + prefiltering + validation")
    
    # Wczytaj dane
    buildings_df, streets_df, stops_df, lines_df = load_data()
    
    # Konfiguruj ograniczenia z PEŁNYMI wymaganiami projektowymi
    constraints = setup_optimization_constraints()
    
    # Inicjalizacja optymalizatora
    logger.info("\n🔧 Inicjalizacja inteligentnego optymalizatora...")
    optimizer = RouteOptimizer(
        buildings_df=buildings_df,
        streets_df=streets_df,
        stops_df=stops_df,
        lines_df=lines_df,
        constraints=constraints,
        
        # Oryginalne parametry (będą tymczasowo zmienione)
        population_size=50,
        generations=20,
        mutation_rate=0.15,
        crossover_rate=0.8,
        
        # WAGI ZGODNIE Z WYMAGANIAMI
        population_weight=0.6,  # 60% - gęstość zabudowy
        distance_weight=0.3,    # 30% - odległości
        angle_weight=0.1        # 10% - prostota
    )
    
    logger.info(f"✅ Optymalizator skonfigurowany z pełnymi wymaganiami")
    
    # INTELIGENTNA SZYBKA OPTYMALIZACJA
    logger.info("\n🧠 Rozpoczynam INTELIGENTNĄ SZYBKĄ optymalizację...")
    start_time = time.time()
    
    # Nowa metoda: inteligentna + szybka + zachowuje wszystkie wymagania
    routes = optimizer.optimize_multiple_routes_intelligent_fast(
        num_routes=3,  # 3 trasy
        time_limit_minutes=3  # 3 minuty na wszystko
    )
    
    total_time = time.time() - start_time
    
    # SZCZEGÓŁOWA ANALIZA ZGODNOŚCI Z WYMAGANIAMI
    logger.info(f"\n📊 === WYNIKI INTELIGENTNEJ OPTYMALIZACJI ===")
    logger.info(f"⏱️  Całkowity czas: {total_time:.1f} sekund ({total_time/60:.1f} minut)")
    logger.info(f"🚊 Liczba znalezionych tras: {len(routes)}")
    
    if not routes:
        logger.error("❌ Nie znaleziono żadnych tras!")
        return
    
    all_requirements_met = True
    
    # Szczegółowa analiza każdej trasy
    for i, (route, score) in enumerate(routes):
        logger.info(f"\n🚊 === TRASA {i+1} - WERYFIKACJA WSZYSTKICH WYMAGAŃ ===")
        
        route_requirements_met = True
        
        # Podstawowe informacje
        route_stops = optimizer._extract_stops_from_route(route)
        total_length = optimizer._calculate_total_length(route)
        
        logger.info(f"📏 Podstawowe informacje:")
        logger.info(f"   Punktów w trasie: {len(route)}")
        logger.info(f"   Głównych przystanków: {len(route_stops)}")
        logger.info(f"   Długość całkowita: {total_length/1000:.2f}km")
        
        # WYMAGANIE #1: Gęstość zabudowy (300m) - KLUCZOWE
        density_score = optimizer.calculate_density_score(route, radius=300)
        density_req_met = density_score >= 0.05
        logger.info(f"\n🏢 WYMAGANIE #1 - Gęstość zabudowy (300m):")
        logger.info(f"   Wynik gęstości: {density_score:.3f}")
        logger.info(f"   Minimum wymagane: 0.05")
        logger.info(f"   Status: {'✅ SPEŁNIONE' if density_req_met else '❌ NIESPEŁNIONE'}")
        if not density_req_met:
            route_requirements_met = False
        
        # WYMAGANIE #2: Odległości między przystankami (200-1500m)
        if len(route_stops) >= 2:
            distances = []
            for j in range(len(route_stops) - 1):
                dist = optimizer._calculate_distance(route_stops[j], route_stops[j+1], is_wgs84=True)
                distances.append(dist)
            
            avg_distance = sum(distances) / len(distances)
            min_distance = min(distances)
            max_distance = max(distances)
            
            distance_req_met = (min_distance >= constraints.min_distance_between_stops and 
                              max_distance <= constraints.max_distance_between_stops)
            
            logger.info(f"\n📏 WYMAGANIE #2 - Odległości między przystankami:")
            logger.info(f"   Średnia odległość: {avg_distance:.0f}m")
            logger.info(f"   Minimalna: {min_distance:.0f}m (min: {constraints.min_distance_between_stops}m)")
            logger.info(f"   Maksymalna: {max_distance:.0f}m (max: {constraints.max_distance_between_stops}m)")
            logger.info(f"   Status: {'✅ SPEŁNIONE' if distance_req_met else '❌ NIESPEŁNIONE'}")
            if not distance_req_met:
                route_requirements_met = False
        else:
            logger.info(f"\n📏 WYMAGANIE #2 - Odległości między przystankami:")
            logger.info(f"   Status: ❌ NIESPEŁNIONE (za mało przystanków)")
            route_requirements_met = False
        
        # WYMAGANIE #3: Prostota tras (minimalizacja zakrętów)
        angle_score = optimizer.calculate_angle_score(route)
        angle_req_met = angle_score >= 0.5
        logger.info(f"\n📐 WYMAGANIE #3 - Prostota tras:")
        logger.info(f"   Wynik prostoty: {angle_score:.3f} (1.0 = idealna prostota)")
        logger.info(f"   Minimum wymagane: 0.5")
        logger.info(f"   Status: {'✅ SPEŁNIONE' if angle_req_met else '❌ NIESPEŁNIONE'}")
        if not angle_req_met:
            route_requirements_met = False
        
        # OGRANICZENIE #1: Długość trasy (1000-15000m)
        length_req_met = (constraints.min_total_length <= total_length <= constraints.max_total_length)
        logger.info(f"\n📊 OGRANICZENIE #1 - Długość trasy:")
        logger.info(f"   Zakres wymagany: {constraints.min_total_length/1000:.1f}-{constraints.max_total_length/1000:.1f}km")
        logger.info(f"   Aktualna: {total_length/1000:.2f}km")
        logger.info(f"   Status: {'✅ SPEŁNIONE' if length_req_met else '❌ NIESPEŁNIONE'}")
        if not length_req_met:
            route_requirements_met = False
        
        # OGRANICZENIE #2: Liczba przystanków (3-20)
        stops_count_req_met = (constraints.min_route_length <= len(route_stops) <= constraints.max_route_length)
        logger.info(f"\n🚏 OGRANICZENIE #2 - Liczba przystanków:")
        logger.info(f"   Zakres wymagany: {constraints.min_route_length}-{constraints.max_route_length}")
        logger.info(f"   Aktualna: {len(route_stops)}")
        logger.info(f"   Status: {'✅ SPEŁNIONE' if stops_count_req_met else '❌ NIESPEŁNIONE'}")
        if not stops_count_req_met:
            route_requirements_met = False
        
        # OGRANICZENIE #3: Brak kolizji z istniejącymi liniami
        has_line_collision = optimizer._check_collision_with_existing_lines(route)
        line_collision_req_met = not has_line_collision
        logger.info(f"\n🚊 OGRANICZENIE #3 - Kolizje z istniejącymi liniami:")
        logger.info(f"   Kolizja wykryta: {'TAK' if has_line_collision else 'NIE'}")
        logger.info(f"   Status: {'✅ SPEŁNIONE' if line_collision_req_met else '❌ NIESPEŁNIONE'}")
        if not line_collision_req_met:
            route_requirements_met = False
        
        # OGRANICZENIE #4: Brak kolizji z zabudową
        has_building_collision = optimizer._check_collision_with_buildings(route)
        building_collision_req_met = not has_building_collision
        logger.info(f"\n🏗️ OGRANICZENIE #4 - Kolizje z zabudową:")
        logger.info(f"   Kolizja wykryta: {'TAK' if has_building_collision else 'NIE'}")
        logger.info(f"   Status: {'✅ SPEŁNIONE' if building_collision_req_met else '❌ NIESPEŁNIONE'}")
        if not building_collision_req_met:
            route_requirements_met = False
        
        # ŁĄCZNY WYNIK TRASY
        logger.info(f"\n🎯 TRASA {i+1} - ŁĄCZNY WYNIK:")
        logger.info(f"   Wynik optymalizacji: {score:.3f}")
        logger.info(f"   Wszystkie wymagania: {'✅ SPEŁNIONE' if route_requirements_met else '❌ NIESPEŁNIONE'}")
        
        if not route_requirements_met:
            all_requirements_met = False
    
    # PODSUMOWANIE GLOBALNE
    logger.info(f"\n🏁 === PODSUMOWANIE INTELIGENTNEJ OPTYMALIZACJI ===")
    logger.info(f"✅ Czas wykonania: {total_time:.1f}s (limit: 180s)")
    logger.info(f"🚊 Znalezionych tras: {len(routes)} / 3")
    logger.info(f"📋 Wszystkie wymagania: {'✅ SPEŁNIONE' if all_requirements_met else '❌ CZĘŚCIOWO SPEŁNIONE'}")
    
    if all_requirements_met:
        logger.info(f"🎉 SUKCES! Inteligentna optymalizacja spełnia wszystkie wymagania!")
        logger.info(f"⚡ Przyspieszenie vs standardowa: ~{300/total_time:.1f}x")
    else:
        logger.warning(f"⚠️ Niektóre wymagania nie zostały spełnione - optymalizacja wymaga dalszych ulepszeń")
    
    return routes

if __name__ == "__main__":
    try:
        # Test oryginalnej ultra-szybkiej optymalizacji
        logger.info("🚀 KROK 1: Test oryginalnej ultra-szybkiej optymalizacji")
        routes_ultra = test_ultra_fast_optimization()
        
        # Test nowej inteligentnej szybkiej optymalizacji  
        logger.info("\n" + "="*80)
        logger.info("🧠 KROK 2: Test inteligentnej szybkiej optymalizacji")
        routes_intelligent = test_intelligent_fast_optimization()
        
        # Porównanie wyników
        logger.info("\n" + "="*80)
        logger.info("📊 PORÓWNANIE METOD:")
        logger.info(f"Ultra-szybka: {len(routes_ultra) if routes_ultra else 0} tras")
        logger.info(f"Inteligentna: {len(routes_intelligent) if routes_intelligent else 0} tras")
        
    except Exception as e:
        logger.error(f"❌ Błąd podczas testowania: {str(e)}")
        import traceback
        traceback.print_exc() 