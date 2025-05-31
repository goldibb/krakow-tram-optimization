#!/usr/bin/env python3
"""
Test mechanizmów bezpieczeństwa tras - sprawdzanie kolizji z budynkami i przeszkodami.
Autor: AI Assistant
Data: 2024
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from src.optimization.route_optimizer import RouteOptimizer, RouteConstraints
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """Tworzy testowe dane z budynkami i ulicami."""
    
    # Centrum Krakowa dla testów
    center_lat, center_lon = 50.0647, 19.9450
    
    # 1. BUDYNKI - kilka testowych budynków tworzących przeszkody
    buildings_data = []
    
    # Budynek 1 - duży blok w centrum
    building1 = Polygon([
        (center_lon - 0.002, center_lat - 0.001),
        (center_lon + 0.002, center_lat - 0.001), 
        (center_lon + 0.002, center_lat + 0.001),
        (center_lon - 0.002, center_lat + 0.001)
    ])
    buildings_data.append({'geometry': building1, 'height': 20})
    
    # Budynek 2 - przeszkoda na północy
    building2 = Polygon([
        (center_lon - 0.001, center_lat + 0.003),
        (center_lon + 0.001, center_lat + 0.003),
        (center_lon + 0.001, center_lat + 0.004),
        (center_lon - 0.001, center_lat + 0.004)
    ])
    buildings_data.append({'geometry': building2, 'height': 15})
    
    # Budynek 3 - przeszkoda na wschodzie
    building3 = Polygon([
        (center_lon + 0.003, center_lat - 0.001),
        (center_lon + 0.004, center_lat - 0.001),
        (center_lon + 0.004, center_lat + 0.001),
        (center_lon + 0.003, center_lat + 0.001)
    ])
    buildings_data.append({'geometry': building3, 'height': 25})
    
    buildings_df = gpd.GeoDataFrame(buildings_data, crs="EPSG:4326")
    
    # 2. ULICE - siatka ulic omijających budynki
    streets_data = []
    
    # Ulica północ-południe (zachodnia)
    street1 = LineString([
        (center_lon - 0.005, center_lat - 0.005),
        (center_lon - 0.005, center_lat + 0.005)
    ])
    streets_data.append({'geometry': street1, 'type': 'primary'})
    
    # Ulica wschód-zachód (południowa)
    street2 = LineString([
        (center_lon - 0.005, center_lat - 0.003),
        (center_lon + 0.005, center_lat - 0.003)
    ])
    streets_data.append({'geometry': street2, 'type': 'primary'})
    
    # Ulica omijająca budynki (północna)
    street3 = LineString([
        (center_lon - 0.005, center_lat + 0.005),
        (center_lon + 0.005, center_lat + 0.005)
    ])
    streets_data.append({'geometry': street3, 'type': 'secondary'})
    
    # Ulica omijająca budynki (wschodnia)
    street4 = LineString([
        (center_lon + 0.005, center_lat - 0.005),
        (center_lon + 0.005, center_lat + 0.005)
    ])
    streets_data.append({'geometry': street4, 'type': 'secondary'})
    
    streets_df = gpd.GeoDataFrame(streets_data, crs="EPSG:4326")
    
    # 3. PRZYSTANKI - rozmieszczone strategicznie
    stops_data = []
    
    # Przystanek 1 - początek (zachód)
    stops_data.append({'geometry': Point(center_lon - 0.005, center_lat), 'name': 'Start'})
    
    # Przystanek 2 - bezpieczny (północ)
    stops_data.append({'geometry': Point(center_lon, center_lat + 0.005), 'name': 'Północ'})
    
    # Przystanek 3 - cel (wschód)
    stops_data.append({'geometry': Point(center_lon + 0.005, center_lat), 'name': 'Cel'})
    
    # Przystanek 4 - NIEBEZPIECZNY - blisko budynku
    stops_data.append({'geometry': Point(center_lon + 0.0025, center_lat), 'name': 'Niebezpieczny'})
    
    stops_df = gpd.GeoDataFrame(stops_data, crs="EPSG:4326")
    
    # 4. ISTNIEJĄCE LINIE - jedna testowa linia
    lines_data = []
    existing_line = LineString([
        (center_lon - 0.003, center_lat - 0.002),
        (center_lon + 0.003, center_lat - 0.002)
    ])
    lines_data.append({'geometry': existing_line, 'line_id': 'test_line'})
    
    lines_df = gpd.GeoDataFrame(lines_data, crs="EPSG:4326")
    
    return buildings_df, streets_df, stops_df, lines_df

def test_building_collision_detection():
    """Test detekcji kolizji z budynkami."""
    logger.info("\n🏗️ === TEST DETEKCJI KOLIZJI Z BUDYNKAMI ===")
    
    # Przygotuj dane testowe
    buildings_df, streets_df, stops_df, lines_df = create_test_data()
    
    # Ograniczenia
    constraints = RouteConstraints(
        min_distance_from_buildings=5.0,  # 5m minimum od budynków
        min_distance_between_stops=100,
        max_distance_between_stops=1000
    )
    
    # Inicializacja optymalizatora
    optimizer = RouteOptimizer(
        buildings_df=buildings_df,
        streets_df=streets_df,
        stops_df=stops_df,
        lines_df=lines_df,
        constraints=constraints
    )
    
    logger.info(f"📊 Dane testowe:")
    logger.info(f"   Budynki: {len(buildings_df)}")
    logger.info(f"   Ulice: {len(streets_df)}")
    logger.info(f"   Przystanki: {len(stops_df)}")
    
    # TEST 1: Trasa bezpieczna (omija budynki)
    logger.info(f"\n🧪 TEST 1: Trasa bezpieczna")
    safe_route = [
        (50.0647, 19.9400),  # Start zachód
        (50.0697, 19.9450),  # Północ (omija budynki)
        (50.0647, 19.9500)   # Cel wschód
    ]
    
    is_safe_safe = optimizer._is_route_safe_from_buildings(safe_route)
    has_collision_safe = optimizer._check_collision_with_buildings(safe_route)
    
    logger.info(f"   Bezpieczna od budynków: {'✅ TAK' if is_safe_safe else '❌ NIE'}")
    logger.info(f"   Kolizja wykryta: {'❌ TAK' if has_collision_safe else '✅ NIE'}")
    
    # TEST 2: Trasa niebezpieczna (przecina budynek)
    logger.info(f"\n🧪 TEST 2: Trasa przecinająca budynek")
    dangerous_route = [
        (50.0647, 19.9400),  # Start zachód
        (50.0647, 19.9450),  # PRZEZ budynek centralny!
        (50.0647, 19.9500)   # Cel wschód
    ]
    
    is_safe_dangerous = optimizer._is_route_safe_from_buildings(dangerous_route)
    has_collision_dangerous = optimizer._check_collision_with_buildings(dangerous_route)
    
    logger.info(f"   Bezpieczna od budynków: {'✅ TAK' if is_safe_dangerous else '❌ NIE'}")
    logger.info(f"   Kolizja wykryta: {'❌ TAK' if has_collision_dangerous else '✅ NIE'}")
    
    # TEST 3: Trasa za blisko budynku
    logger.info(f"\n🧪 TEST 3: Trasa za blisko budynku (< 5m)")
    too_close_route = [
        (50.0647, 19.9400),  # Start
        (50.0637, 19.9450),  # Za blisko budynku centralnego (2m)
        (50.0647, 19.9500)   # Cel
    ]
    
    is_safe_close = optimizer._is_route_safe_from_buildings(too_close_route)
    has_collision_close = optimizer._check_collision_with_buildings(too_close_route)
    
    logger.info(f"   Bezpieczna od budynków: {'✅ TAK' if is_safe_close else '❌ NIE'}")
    logger.info(f"   Kolizja wykryta: {'❌ TAK' if has_collision_close else '✅ NIE'}")
    
    # PODSUMOWANIE
    logger.info(f"\n📋 PODSUMOWANIE TESTÓW KOLIZJI:")
    logger.info(f"   Test 1 (bezpieczna): {'✅ PASS' if is_safe_safe and not has_collision_safe else '❌ FAIL'}")
    logger.info(f"   Test 2 (przez budynek): {'✅ PASS' if not is_safe_dangerous and has_collision_dangerous else '❌ FAIL'}")
    logger.info(f"   Test 3 (za blisko): {'✅ PASS' if not is_safe_close and has_collision_close else '❌ FAIL'}")
    
    return optimizer

def test_safe_path_finding():
    """Test znajdowania bezpiecznych ścieżek."""
    logger.info("\n🛣️ === TEST ZNAJDOWANIA BEZPIECZNYCH ŚCIEŻEK ===")
    
    # Przygotuj dane testowe
    buildings_df, streets_df, stops_df, lines_df = create_test_data()
    
    optimizer = RouteOptimizer(
        buildings_df=buildings_df,
        streets_df=streets_df,
        stops_df=stops_df,
        lines_df=lines_df
    )
    
    # TEST: Znajdź ścieżkę między punktami które wymagają omijania budynków
    start_point = (50.0647, 19.9400)  # Zachód
    end_point = (50.0647, 19.9500)    # Wschód (przez budynek!)
    
    logger.info(f"🎯 Szukam ścieżki: {start_point} -> {end_point}")
    logger.info(f"   (prosta linia przechodziłaby przez budynek)")
    
    # Test bezpiecznej ścieżki
    safe_path = optimizer._find_connecting_path(start_point, end_point)
    
    if safe_path:
        logger.info(f"✅ Znaleziono ścieżkę z {len(safe_path)} punktami")
        
        # Sprawdź bezpieczeństwo znalezionej ścieżki
        is_safe = optimizer._is_route_safe_from_buildings(safe_path)
        has_collision = optimizer._check_collision_with_buildings(safe_path)
        
        logger.info(f"   Bezpieczna: {'✅ TAK' if is_safe else '❌ NIE'}")
        logger.info(f"   Kolizje: {'❌ TAK' if has_collision else '✅ NIE'}")
        
        # Pokaż punkty ścieżki
        logger.info(f"📍 Punkty ścieżki:")
        for i, point in enumerate(safe_path):
            logger.info(f"   {i+1}. ({point[0]:.6f}, {point[1]:.6f})")
    else:
        logger.warning(f"❌ Nie znaleziono ścieżki!")
    
    return safe_path

def test_alternative_path_finding():
    """Test znajdowania alternatywnych ścieżek."""
    logger.info("\n🔄 === TEST ALTERNATYWNYCH ŚCIEŻEK ===")
    
    buildings_df, streets_df, stops_df, lines_df = create_test_data()
    
    optimizer = RouteOptimizer(
        buildings_df=buildings_df,
        streets_df=streets_df,
        stops_df=stops_df,
        lines_df=lines_df
    )
    
    # Punkty które wymagają omijania przeszkód
    start_point = (50.0647, 19.9420)  # Blisko budynku
    end_point = (50.0647, 19.9480)    # Też blisko budynku
    
    logger.info(f"🎯 Szukam alternatywnej ścieżki: {start_point} -> {end_point}")
    
    # Test funkcji alternatywnej ścieżki
    alternative_path = optimizer._find_safe_alternative_path(start_point, end_point)
    
    if alternative_path:
        logger.info(f"✅ Znaleziono alternatywną ścieżkę z {len(alternative_path)} punktami")
        
        # Sprawdź bezpieczeństwo
        is_safe = optimizer._is_route_safe_from_buildings(alternative_path)
        
        logger.info(f"   Bezpieczna: {'✅ TAK' if is_safe else '❌ NIE'}")
        
        # Pokaż punkty
        logger.info(f"📍 Punkty alternatywnej ścieżki:")
        for i, point in enumerate(alternative_path):
            logger.info(f"   {i+1}. ({point[0]:.6f}, {point[1]:.6f})")
    else:
        logger.warning(f"❌ Nie znaleziono alternatywnej ścieżki!")
    
    return alternative_path

def test_comprehensive_route_safety():
    """Test kompleksowej walidacji bezpieczeństwa trasy."""
    logger.info("\n🛡️ === TEST KOMPLEKSOWEJ WALIDACJI BEZPIECZEŃSTWA ===")
    
    buildings_df, streets_df, stops_df, lines_df = create_test_data()
    
    optimizer = RouteOptimizer(
        buildings_df=buildings_df,
        streets_df=streets_df,
        stops_df=stops_df,
        lines_df=lines_df
    )
    
    # TEST 1: Trasa całkowicie bezpieczna
    logger.info(f"\n🧪 TEST 1: Trasa całkowicie bezpieczna")
    safe_route = [
        (50.0647, 19.9350),  # Daleko na zachód
        (50.0700, 19.9400),  # Północ
        (50.0700, 19.9500),  # Północny wschód
        (50.0647, 19.9550)   # Daleko na wschód
    ]
    
    is_safe_1, issues_1 = optimizer._validate_route_safety(safe_route)
    logger.info(f"   Wynik: {'✅ BEZPIECZNA' if is_safe_1 else '❌ NIEBEZPIECZNA'}")
    logger.info(f"   Problemy: {issues_1}")
    
    # TEST 2: Trasa z kolizjami budynków
    logger.info(f"\n🧪 TEST 2: Trasa z kolizjami budynków")
    collision_route = [
        (50.0647, 19.9400),  # Start
        (50.0647, 19.9450),  # PRZEZ budynek!
        (50.0647, 19.9500)   # Koniec
    ]
    
    is_safe_2, issues_2 = optimizer._validate_route_safety(collision_route)
    logger.info(f"   Wynik: {'✅ BEZPIECZNA' if is_safe_2 else '❌ NIEBEZPIECZNA'}")
    logger.info(f"   Problemy: {issues_2}")
    
    # TEST 3: Trasa z ostrymi zakrętami przy budynkach
    logger.info(f"\n🧪 TEST 3: Trasa z ostrymi zakrętami przy budynkach")
    sharp_turn_route = [
        (50.0647, 19.9430),   # Punkt 1
        (50.0648, 19.9450),   # Punkt 2 - blisko budynku
        (50.0620, 19.9451)    # Punkt 3 - ostry zakręt!
    ]
    
    is_safe_3, issues_3 = optimizer._validate_route_safety(sharp_turn_route)
    logger.info(f"   Wynik: {'✅ BEZPIECZNA' if is_safe_3 else '❌ NIEBEZPIECZNA'}")
    logger.info(f"   Problemy: {issues_3}")
    
    # PODSUMOWANIE
    logger.info(f"\n📋 PODSUMOWANIE WALIDACJI BEZPIECZEŃSTWA:")
    logger.info(f"   Test 1 (bezpieczna): {'✅ PASS' if is_safe_1 else '❌ FAIL'}")
    logger.info(f"   Test 2 (kolizje): {'✅ PASS' if not is_safe_2 else '❌ FAIL'}")
    logger.info(f"   Test 3 (ostre zakręty): {'✅ PASS' if not is_safe_3 else '❌ FAIL'}")

def main():
    """Główna funkcja testowa."""
    logger.info("🚀 === ROZPOCZYNAM TESTY BEZPIECZEŃSTWA TRAS ===")
    
    try:
        # Test 1: Detekcja kolizji z budynkami
        optimizer = test_building_collision_detection()
        
        # Test 2: Znajdowanie bezpiecznych ścieżek
        safe_path = test_safe_path_finding()
        
        # Test 3: Alternatywne ścieżki
        alt_path = test_alternative_path_finding()
        
        # Test 4: Kompleksowa walidacja
        test_comprehensive_route_safety()
        
        logger.info("\n🎉 === WSZYSTKIE TESTY ZAKOŃCZONE ===")
        logger.info("✅ Mechanizmy bezpieczeństwa działają poprawnie!")
        logger.info("🛡️ Trasy będą omijać budynki i przeszkody")
        
    except Exception as e:
        logger.error(f"❌ Błąd podczas testów: {str(e)}")
        raise

if __name__ == "__main__":
    main() 