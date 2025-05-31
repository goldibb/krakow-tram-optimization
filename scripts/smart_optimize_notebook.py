#!/usr/bin/env python3
"""
Skrypt do notebooka - Optymalizacja tras tramwajowych w Krakowie
Nowy, inteligentny algorytm zgodny z wymaganiami hackathonu.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Dodaj ścieżkę do modułów
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium import plugins
import logging
from typing import List, Tuple, Dict

# Import nowego optymalizatora
from src.optimization.smart_route_optimizer import SmartRouteOptimizer, RouteConstraints
from src.visualization.route_visualizer import RouteVisualizer

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_krakow_data(data_dir: str = None) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Wczytuje dane o Krakowie z plików GeoJSON.
    
    Args:
        data_dir: Ścieżka do katalogu z danymi (domyślnie ../data)
        
    Returns:
        Tuple z DataFrames: buildings, streets, stops, lines
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    print("🗂️ Wczytywanie danych o Krakowie...")
    
    # Ścieżki do plików
    files = {
        'buildings': os.path.join(data_dir, 'buildings.geojson'),
        'streets': os.path.join(data_dir, 'streets.geojson'),
        'stops': os.path.join(data_dir, 'stops.geojson'),
        'lines': os.path.join(data_dir, 'lines.geojson')
    }
    
    # Sprawdź czy pliki istnieją
    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Nie znaleziono pliku: {path}")
    
    # Wczytaj dane
    buildings_df = gpd.read_file(files['buildings'])
    streets_df = gpd.read_file(files['streets'])
    stops_df = gpd.read_file(files['stops'])
    lines_df = gpd.read_file(files['lines'])
    
    print(f"✅ Wczytano dane:")
    print(f"   📘 Budynki: {len(buildings_df):,}")
    print(f"   🛣️ Ulice: {len(streets_df):,}")
    print(f"   🚏 Przystanki: {len(stops_df):,}")
    print(f"   🚊 Linie tramwajowe: {len(lines_df):,}")
    
    return buildings_df, streets_df, stops_df, lines_df

def setup_optimizer(buildings_df: gpd.GeoDataFrame, 
                   streets_df: gpd.GeoDataFrame, 
                   stops_df: gpd.GeoDataFrame, 
                   lines_df: gpd.GeoDataFrame) -> SmartRouteOptimizer:
    """
    Konfiguruje optymalizator tras zgodnie z wymaganiami hackathonu.
    
    Returns:
        Skonfigurowany SmartRouteOptimizer
    """
    print("⚙️ Konfiguracja optymalizatora...")
    
    # Ograniczenia zgodne z wymaganiami hackathonu
    constraints = RouteConstraints(
        min_distance_between_stops=350,    # min 350m między przystankami
        max_distance_between_stops=700,    # max 700m między przystankami  
        min_total_length=1500,             # min 1.5km długość trasy
        max_total_length=15000,            # max 15km długość trasy
        min_route_stops=4,                 # min 4 przystanki
        max_route_stops=15,                # max 15 przystanków
        min_distance_from_buildings=5.0,   # 5m od budynków
        buffer_around_existing_lines=50.0  # 50m od istniejących linii
    )
    
    # Inicjalizuj optymalizator
    optimizer = SmartRouteOptimizer(
        buildings_df=buildings_df,
        streets_df=streets_df,
        stops_df=stops_df,
        lines_df=lines_df,
        constraints=constraints
    )
    
    print("✅ Optymalizator skonfigurowany zgodnie z wymaganiami hackathonu")
    return optimizer

def optimize_tram_routes(optimizer: SmartRouteOptimizer, 
                        num_routes: int = 3) -> List[Tuple[List[Tuple[float, float]], float]]:
    """
    Optymalizuje trasy tramwajowe zgodnie z wymaganiami:
    1. Maksymalizuje pokrycie obszarów o dużej gęstości zabudowy
    2. Maksymalizuje dystans między przystankami  
    3. Minimalizuje liczbę zakrętów
    4. Unika kolizji z budynkami
    5. Nie pokrywa się z istniejącymi liniami
    
    Args:
        optimizer: Skonfigurowany optymalizator
        num_routes: Liczba tras do optymalizacji
        
    Returns:
        Lista par (trasa, ocena)
    """
    print(f"🚊 Rozpoczynam optymalizację {num_routes} tras tramwajowych...")
    print("📋 Wymagania:")
    print("   ✓ Maksymalizacja pokrycia obszarów o dużej gęstości zabudowy")
    print("   ✓ Maksymalizacja dystansu między przystankami")
    print("   ✓ Minimalizacja liczby zakrętów")
    print("   ✓ Unikanie kolizji z budynkami")
    print("   ✓ Brak pokrywania z istniejącymi liniami")
    print("   ✓ Lokalne budowanie tras (bez 'skakania')")
    
    # Uruchom optymalizację
    optimized_routes = optimizer.optimize_routes(
        num_routes=num_routes,
        max_iterations=50  # Zmniejszone dla szybszego działania w notebooku
    )
    
    print(f"\n🎉 Optymalizacja zakończona!")
    print(f"✅ Znaleziono {len(optimized_routes)}/{num_routes} tras")
    
    # Pokaż statystyki
    stats = optimizer.get_optimization_stats()
    print(f"\n📊 Statystyki uczenia się:")
    print(f"   🧠 Zapamiętane trasy: {stats['successful_routes']}")
    print(f"   🔗 Nauczone połączenia: {stats['learned_connections']}")
    print(f"   ❌ Złe obszary: {stats['bad_areas']}")
    
    return optimized_routes

def analyze_routes(routes: List[Tuple[List[Tuple[float, float]], float]], 
                  optimizer: SmartRouteOptimizer) -> None:
    """
    Analizuje znalezione trasy i pokazuje szczegółowe informacje.
    
    Args:
        routes: Lista znalezionych tras
        optimizer: Optymalizator (do obliczeń)
    """
    print(f"\n📋 ANALIZA ZNALEZIONYCH TRAS")
    print("=" * 50)
    
    for i, (route, score) in enumerate(routes, 1):
        print(f"\n🚊 TRASA {i}:")
        print(f"   📊 Ocena: {score:.1f}/100")
        print(f"   🚏 Liczba przystanków: {len(route)}")
        
        # Oblicz długość trasy
        total_length = 0
        for j in range(len(route) - 1):
            dist = optimizer._calculate_distance_wgs84(route[j], route[j + 1])
            total_length += dist
        
        print(f"   📏 Długość trasy: {total_length/1000:.1f} km")
        print(f"   🗺️ Zakres: lat {min(p[0] for p in route):.4f}-{max(p[0] for p in route):.4f}")
        print(f"           lon {min(p[1] for p in route):.4f}-{max(p[1] for p in route):.4f}")
        
        # Sprawdź bezpieczeństwo
        is_safe, safety_msg = optimizer.is_route_safe(route)
        safety_status = "✅ BEZPIECZNA" if is_safe else "❌ PROBLEMY"
        print(f"   🔒 Bezpieczeństwo: {safety_status} - {safety_msg}")
        
        # Oceń gęstość zabudowy
        density_scores = []
        for lat, lon in route:
            density = optimizer.density_calculator.calculate_density_at_point(lat, lon)
            density_scores.append(density)
        
        avg_density = np.mean(density_scores)
        print(f"   🏘️ Średnia gęstość zabudowy: {avg_density:.2f}")
        
        # Sprawdź odległości między przystankami
        distances = []
        for j in range(len(route) - 1):
            dist = optimizer._calculate_distance_wgs84(route[j], route[j + 1])
            distances.append(dist)
        
        if distances:
            print(f"   📐 Odległości: min {min(distances):.0f}m, "
                  f"max {max(distances):.0f}m, śred {np.mean(distances):.0f}m")

def create_interactive_map(routes: List[Tuple[List[Tuple[float, float]], float]], 
                          buildings_df: gpd.GeoDataFrame,
                          stops_df: gpd.GeoDataFrame,
                          lines_df: gpd.GeoDataFrame = None) -> folium.Map:
    """
    Tworzy interaktywną mapę z zoptymalizowanymi trasami.
    
    Args:
        routes: Lista znalezionych tras
        buildings_df: DataFrame z budynkami
        stops_df: DataFrame z przystankami  
        lines_df: DataFrame z istniejącymi liniami
        
    Returns:
        Interaktywna mapa Folium
    """
    print("🗺️ Tworzenie interaktywnej mapy...")
    
    if not routes:
        print("❌ Brak tras do wizualizacji")
        return None
    
    # Centrum mapy - średnia z wszystkich tras
    all_points = []
    for route, _ in routes:
        all_points.extend(route)
    
    center_lat = np.mean([p[0] for p in all_points])
    center_lon = np.mean([p[1] for p in all_points])
    
    # Utwórz mapę
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Kolory dla tras
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
    
    # Dodaj istniejące linie tramwajowe (tło)
    if lines_df is not None:
        print("   Dodawanie istniejących linii tramwajowych...")
        for _, line in lines_df.iterrows():
            if hasattr(line.geometry, 'coords'):
                coords = [(lat, lon) for lon, lat in line.geometry.coords]
                folium.PolyLine(
                    coords, 
                    color='gray', 
                    weight=2, 
                    opacity=0.5,
                    popup="Istniejąca linia tramwajowa"
                ).add_to(m)
    
    # Dodaj przystanki (małe kropki)
    print("   Dodawanie przystanków...")
    for _, stop in stops_df.iterrows():
        folium.CircleMarker(
            location=[stop.geometry.y, stop.geometry.x],
            radius=2,
            color='lightgray',
            fill=True,
            fillOpacity=0.3,
            popup=f"Przystanek"
        ).add_to(m)
    
    # Dodaj nowe trasy
    print("   Dodawanie nowych zoptymalizowanych tras...")
    for i, (route, score) in enumerate(routes):
        color = colors[i % len(colors)]
        
        # Linia trasy
        folium.PolyLine(
            route, 
            color=color, 
            weight=4, 
            opacity=0.8,
            popup=f"Nowa trasa {i+1} (ocena: {score:.1f})"
        ).add_to(m)
        
        # Przystanki na trasie
        for j, (lat, lon) in enumerate(route):
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.8,
                popup=f"Trasa {i+1} - Przystanek {j+1}"
            ).add_to(m)
    
    # Dodaj legendę
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>Legenda</h4>
    <p><span style="color:gray;">●</span> Istniejące linie</p>
    <p><span style="color:lightgray;">●</span> Przystanki</p>
    <p><span style="color:red;">●</span> Nowa trasa 1</p>
    <p><span style="color:blue;">●</span> Nowa trasa 2</p>
    <p><span style="color:green;">●</span> Nowa trasa 3</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    print("✅ Mapa utworzona")
    return m

def save_results(routes: List[Tuple[List[Tuple[float, float]], float]], 
                output_dir: str = None) -> str:
    """
    Zapisuje wyniki optymalizacji do plików.
    
    Args:
        routes: Lista znalezionych tras
        output_dir: Katalog wyjściowy (domyślnie ../results)
        
    Returns:
        Ścieżka do zapisanych plików
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"💾 Zapisywanie wyników do {output_dir}...")
    
    # Zapisz jako CSV
    routes_data = []
    for i, (route, score) in enumerate(routes, 1):
        for j, (lat, lon) in enumerate(route):
            routes_data.append({
                'route_id': i,
                'stop_order': j + 1,
                'latitude': lat,
                'longitude': lon,
                'route_score': score
            })
    
    df = pd.DataFrame(routes_data)
    csv_path = os.path.join(output_dir, 'optimized_routes.csv')
    df.to_csv(csv_path, index=False)
    
    # Zapisz jako GeoJSON
    geometries = []
    for i, (route, score) in enumerate(routes, 1):
        from shapely.geometry import LineString
        line = LineString([(lon, lat) for lat, lon in route])
        geometries.append({
            'geometry': line,
            'route_id': i,
            'score': score,
            'num_stops': len(route)
        })
    
    if geometries:
        gdf = gpd.GeoDataFrame(geometries, crs='EPSG:4326')
        geojson_path = os.path.join(output_dir, 'optimized_routes.geojson')
        gdf.to_file(geojson_path, driver='GeoJSON')
    
    print(f"✅ Zapisano:")
    print(f"   📄 CSV: {csv_path}")
    print(f"   🗺️ GeoJSON: {geojson_path}")
    
    return output_dir

# GŁÓWNA FUNKCJA DO URUCHOMIENIA W NOTEBOOKU
def run_tram_optimization(data_dir: str = None, 
                         num_routes: int = 3,
                         save_map: bool = True) -> Tuple[List, folium.Map]:
    """
    GŁÓWNA FUNKCJA - uruchom pełną optymalizację tras tramwajowych.
    
    Args:
        data_dir: Katalog z danymi (domyślnie ../data)
        num_routes: Liczba tras do optymalizacji (domyślnie 3)
        save_map: Czy zapisać mapę do pliku (domyślnie True)
        
    Returns:
        Tuple (lista_tras, mapa_folium)
        
    Użycie w notebooku:
        routes, map_viz = run_tram_optimization()
        map_viz  # wyświetl mapę
    """
    print("🚊 OPTYMALIZACJA TRAS TRAMWAJOWYCH W KRAKOWIE")
    print("=" * 60)
    print("📋 Zgodne z wymaganiami hackathonu:")
    print("   ✓ Maksymalizacja gęstości zabudowy (300m radius)")
    print("   ✓ Maksymalizacja dystansu między przystankami")  
    print("   ✓ Unikanie kolizji z budynkami (5m buffer)")
    print("   ✓ Brak pokrywania z istniejącymi liniami (50m buffer)")
    print("   ✓ Lokalne budowanie tras bez 'skakania'")
    print("   ✓ Algorytm z uczeniem się")
    print("=" * 60)
    
    try:
        # 1. Wczytaj dane
        buildings_df, streets_df, stops_df, lines_df = load_krakow_data(data_dir)
        
        # 2. Skonfiguruj optymalizator
        optimizer = setup_optimizer(buildings_df, streets_df, stops_df, lines_df)
        
        # 3. Optymalizuj trasy
        routes = optimize_tram_routes(optimizer, num_routes)
        
        if not routes:
            print("❌ Nie znaleziono żadnych tras!")
            return [], None
        
        # 4. Analizuj wyniki
        analyze_routes(routes, optimizer)
        
        # 5. Utwórz mapę
        interactive_map = create_interactive_map(routes, buildings_df, stops_df, lines_df)
        
        # 6. Zapisz wyniki
        results_dir = save_results(routes)
        
        # 7. Zapisz mapę jeśli requested
        if save_map and interactive_map:
            map_path = os.path.join(results_dir, 'interactive_map.html')
            interactive_map.save(map_path)
            print(f"🗺️ Mapa zapisana: {map_path}")
        
        print(f"\n🎉 OPTYMALIZACJA ZAKOŃCZONA POMYŚLNIE!")
        print(f"✅ Znaleziono {len(routes)} tras spełniających wszystkie wymagania")
        
        return routes, interactive_map
        
    except Exception as e:
        print(f"❌ Błąd podczas optymalizacji: {e}")
        import traceback
        traceback.print_exc()
        return [], None

if __name__ == "__main__":
    # Uruchomienie standalone
    routes, map_viz = run_tram_optimization(num_routes=3)
    if map_viz:
        print("Mapa zapisana jako 'interactive_map.html'") 