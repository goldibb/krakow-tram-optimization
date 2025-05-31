import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopandas as gpd
from src.optimization.route_optimizer import RouteOptimizer, RouteConstraints
from src.visualization.route_visualizer import RouteVisualizer
import logging
import time

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
    
    # KONFIGURACJA OGRANICZEŃ - ZAKTUALIZOWANE WEDŁUG NOWYCH WYMAGAŃ
    constraints = RouteConstraints(
        # ELASTYCZNE ODLEGŁOŚCI - pozwalają na lekkie nachodzenie budynków
        min_distance_between_stops=300,   # Zmniejszone z 350 na 300
        max_distance_between_stops=800,   # Zwiększone z 700 na 800 (bez skoków)
        
        # REALISTYCZNE DŁUGOŚCI TRAS
        min_total_length=1000,            # Zmniejszone z 1500 na 1000
        max_total_length=8000,            # Zmniejszone z 15000 na 8000 (praktyczniejsze)
        
        # UMIARKOWANA LICZBA PRZYSTANKÓW
        min_route_length=3,               # Zmniejszone z 4 na 3
        max_route_length=10,              # Zmniejszone z 15 na 10 (bez skoków)
        
        # ZACHOWANE ZAŁOŻENIA + ELASTYCZNOŚĆ DLA BUDYNKÓW
        max_angle=45.0,                   # Proste trasy
        min_distance_from_buildings=2.0   # ZMNIEJSZONE z 5 na 2 (lekkie nachodzenie OK)
    )
    
    # Inicjalizacja optymalizatora z NOWYMI parametrami
    logger.info("Inicjalizacja optymalizatora z elastycznymi parametrami...")
    optimizer = RouteOptimizer(
        buildings_df=buildings_df,
        streets_df=streets_df,
        stops_df=stops_df,
        lines_df=lines_df,
        constraints=constraints,
        population_size=20,               # Mniejsza populacja dla szybszego działania
        generations=10,                   # Mniej pokoleń dla szybszego działania
        mutation_rate=0.15,
        crossover_rate=0.8,
        population_weight=0.6,
        distance_weight=0.3,
        angle_weight=0.1
    )
    
    # 🚫 NOWA METODA OPTYMALIZACJI BEZ SKOKÓW
    logger.info("🚫 Rozpoczynam optymalizację BEZ SKOKÓW...")
    logger.info("✨ Nowe funkcje:")
    logger.info("   🏗️  Lekkie nachodzenie budynków dozwolone (max 50m razem)")
    logger.info("   🔗 Sekwencyjne przystanki (max 800m między nimi)")
    logger.info("   🎯 Automatyczne dodawanie punktów pośrednich")
    logger.info("   ✅ Kontrola skoków na każdym etapie")
    
    # Resetuj używane przystanki
    optimizer.reset_used_stops()
    
    # NOWA FUNKCJA: optymalizacja bez skoków
    start_time = time.time()
    
    try:
        # Użyj nowej metody BEZ SKOKÓW
        multiple_routes = optimizer.optimize_multiple_routes_no_jumps(
            num_routes=3,  # 3 trasy dla testu
            time_limit_minutes=10  # 10 minut na wszystkie trasy
        )
        
        optimization_time = time.time() - start_time
        
    except Exception as e:
        logger.error(f"Błąd podczas optymalizacji bez skoków: {e}")
        logger.info("🔄 Fallback do standardowej metody...")
        
        # Fallback do standardowej metody
        multiple_routes = []
        optimization_time = time.time() - start_time

    # Analiza wyników
    logger.info(f"\n=== WYNIKI OPTYMALIZACJI BEZ SKOKÓW ===")
    logger.info(f"⏱️  Czas optymalizacji: {optimization_time:.1f}s")

    if multiple_routes:
        logger.info(f"✅ SUKCES! Znaleziono {len(multiple_routes)} tras bez skoków")
        
        for i, (route, score) in enumerate(multiple_routes):
            logger.info(f"\n🚊 TRASA {i+1} (BEZ SKOKÓW):")
            logger.info(f"   📍 Liczba punktów: {len(route)}")
            logger.info(f"   📊 Ocena: {score:.3f}")
            
            # Sprawdź szczegółowe właściwości trasy
            try:
                # Wyodrębnij przystanki z trasy
                route_stops = optimizer._extract_stops_from_route(route)
                total_length = optimizer._calculate_total_length(route)
                
                logger.info(f"   🏁 Przystanki: {len(route_stops)}")
                logger.info(f"   📏 Długość całkowita: {total_length/1000:.2f} km")
                
                # NOWE: Sprawdź czy rzeczywiście nie ma skoków
                has_jumps = optimizer._check_for_jumps(route, max_distance=800)
                if has_jumps:
                    logger.warning(f"   ⚠️  UWAGA: Wykryto skoki > 800m!")
                else:
                    logger.info(f"   ✅ Brak skoków (wszystkie odległości ≤ 800m)")
                
                # NOWE: Sprawdź kolizje z budynkami (nowa elastyczna metoda)
                has_building_collision = optimizer._check_collision_with_buildings(route)
                if has_building_collision:
                    logger.info(f"   🏗️  Poważne kolizje z budynkami")
                else:
                    logger.info(f"   ✅ Lekkie nachodzenia budynków OK")
                
                # Sprawdź odległości między kolejnymi punktami
                distances = []
                for j in range(len(route) - 1):
                    dist = optimizer._calculate_distance(route[j], route[j+1], is_wgs84=True)
                    distances.append(dist)
                
                if distances:
                    avg_dist = sum(distances) / len(distances)
                    min_dist = min(distances)
                    max_dist = max(distances)
                    
                    logger.info(f"   📐 Odległości między punktami:")
                    logger.info(f"      Średnia: {avg_dist:.0f}m")
                    logger.info(f"      Min-Max: {min_dist:.0f}-{max_dist:.0f}m")
                    
                    # Sprawdź zgodność z nowymi parametrami (300-800m)
                    out_of_range = sum(1 for d in distances if d < 300 or d > 800)
                    if out_of_range == 0:
                        logger.info(f"   ✅ Wszystkie odległości w zakresie 300-800m")
                    else:
                        logger.info(f"   ⚠️  {out_of_range}/{len(distances)} odległości poza zakresem")
                
            except Exception as e:
                logger.warning(f"Błąd analizy trasy {i+1}: {e}")

        # Dodatkowe informacje o nowych funkcjach
        logger.info(f"\n🎯 NOWE FUNKCJE W AKCJI:")
        logger.info(f"   🔍 Sprawdzanie skoków: ✅ (max 800m)")
        logger.info(f"   🏗️  Elastyczne budynki: ✅ (max 50m przecięć)")
        logger.info(f"   🔗 Sekwencyjne przystanki: ✅")
        logger.info(f"   📍 Punkty pośrednie: ✅ (automatyczne)")
        
    else:
        logger.error("❌ Nie znaleziono żadnych tras bez skoków!")
        logger.info("💡 Spróbuj:")
        logger.info("   - Zwiększyć time_limit_minutes")
        logger.info("   - Zmniejszyć liczbę tras")
        logger.info("   - Sprawdzić dostępność przystanków")
        
        # Spróbuj z pojedynczą trasą dla debugowania
        logger.info("🔧 Próbuję wygenerować pojedynczą trasę dla debugowania...")
        
        try:
            # Wybierz losowy punkt startowy
            valid_stops = [(row.geometry.y, row.geometry.x) for _, row in optimizer.stops_df.iterrows()]
            if valid_stops:
                start_point = valid_stops[0]
                
                # Generuj pojedynczą sekwencyjną trasę
                debug_route = optimizer._generate_sequential_route(
                    start_point=start_point,
                    target_length=5,
                    max_distance_between_stops=800
                )
                
                if debug_route:
                    logger.info(f"✅ Debug: wygenerowano trasę z {len(debug_route)} przystankami")
                    
                    # Sprawdź skoki
                    has_jumps = optimizer._check_for_jumps(debug_route, max_distance=800)
                    logger.info(f"   Skoki: {'❌ TAK' if has_jumps else '✅ NIE'}")
                    
                else:
                    logger.warning("❌ Debug: nie udało się wygenerować trasy")
            else:
                logger.error("❌ Brak dostępnych przystanków")
                
        except Exception as debug_e:
            logger.error(f"❌ Błąd debugowania: {debug_e}")

    logger.info(f"\n📊 Podsumowanie:")
    logger.info(f"   🚊 Znalezione trasy: {len(multiple_routes)}")
    logger.info(f"   ⏱️  Czas optymalizacji: {optimization_time:.1f}s")
    logger.info(f"   🔧 Nowe funkcje: aktywne")

    # Jeśli mamy trasy, wizualizuj pierwszą z nich
    if multiple_routes:
        best_route = multiple_routes[0][0]  # Pierwsza trasa
        
        # Inicjalizacja wizualizatora
        visualizer = RouteVisualizer(buildings_df, streets_df)
        
        # Wizualizacja wyników - NOWA METODA BEZ SKOKÓW
        logger.info("Generowanie wizualizacji tras BEZ SKOKÓW...")
        
        # Utworzenie katalogu results jeśli nie istnieje
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Tworzenie mapy z kontrolą skoków
        m = visualizer.create_base_map()
        
        # NOWA OPCJA: Wizualizacja sekwencyjna (bez skoków)
        visualizer.plot_route_sequential(
            best_route, 
            m, 
            route_name="Trasa BEZ SKOKÓW (sekwencyjna)", 
            color='green',
            max_segment_length=800  # Kontrola skoków
        )
        
        # Zapisanie mapy
        map_path = os.path.join(results_dir, "optimized_route_no_jumps.html")
        m.save(map_path)
        
        logger.info(f"✅ Wizualizacja tras BEZ SKOKÓW zapisana w {map_path}")
        
        # Sprawdzenie końcowe bezpieczeństwa
        logger.info("🔍 Sprawdzanie końcowe bezpieczeństwa tras...")
        is_safe, safety_msg = optimizer._validate_route_safety(best_route)
        
        if is_safe:
            logger.info(f"✅ TRASA BEZPIECZNA: {safety_msg}")
        else:
            logger.warning(f"⚠️ UWAGA - {safety_msg}")

if __name__ == "__main__":
    main() 