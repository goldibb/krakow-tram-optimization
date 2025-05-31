import geopandas as gpd
import pandas as pd
import numpy as np
import os
from shapely.geometry import Point, LineString

def analyze_tram_data():
    """Analizuje istniejące dane tramwajowe w Krakowie."""
    
    print("🚊 === ANALIZA ISTNIEJĄCYCH DANYCH TRAMWAJOWYCH KRAKÓW ===")
    
    # Wczytaj dane
    data_dir = 'data'
    buildings_df = gpd.read_file(os.path.join(data_dir, 'buildings.geojson'))
    streets_df = gpd.read_file(os.path.join(data_dir, 'streets.geojson'))
    stops_df = gpd.read_file(os.path.join(data_dir, 'stops.geojson'))
    lines_df = gpd.read_file(os.path.join(data_dir, 'lines.geojson'))
    
    print(f"📊 Podstawowe statystyki:")
    print(f"   Budynki: {len(buildings_df)}")
    print(f"   Ulice: {len(streets_df)}")
    print(f"   Przystanki: {len(stops_df)}")
    print(f"   Linie tramwajowe: {len(lines_df)}")
    
    # ANALIZA 1: Struktura danych linii
    print(f"\n🔍 ANALIZA STRUKTURY DANYCH:")
    print(f"Kolumny w lines_df: {list(lines_df.columns)}")
    print(f"Kolumny w stops_df: {list(stops_df.columns)}")
    
    # Sprawdź przykładowe dane
    print(f"\n📋 Przykładowe linie tramwajowe:")
    for i, row in lines_df.head().iterrows():
        line_info = []
        for col in lines_df.columns:
            if col != 'geometry':
                line_info.append(f"{col}: {row[col]}")
        print(f"   Linia {i}: {', '.join(line_info)}")
    
    # ANALIZA 2: Długości linii tramwajowych
    print(f"\n📏 ANALIZA DŁUGOŚCI LINII TRAMWAJOWYCH:")
    
    # Konwertuj do EPSG:2180 dla dokładnych obliczeń w metrach
    lines_projected = lines_df.to_crs(epsg=2180)
    
    # Oblicz długości tras
    line_lengths = []
    line_details = []
    
    for i, row in lines_projected.iterrows():
        if isinstance(row.geometry, LineString):
            length_m = row.geometry.length
            length_km = length_m / 1000
            line_lengths.append(length_km)
            
            # Przygotuj szczegóły linii
            line_detail = {
                'index': i,
                'length_km': length_km,
                'coords_count': len(list(row.geometry.coords))
            }
            
            # Dodaj inne dostępne informacje
            for col in lines_df.columns:
                if col != 'geometry':
                    line_detail[col] = row[col] if col in row and pd.notna(row[col]) else 'N/A'
            
            line_details.append(line_detail)
    
    if line_lengths:
        print(f"   📈 Statystyki długości tras:")
        print(f"      Minimalna długość: {min(line_lengths):.2f} km")
        print(f"      Maksymalna długość: {max(line_lengths):.2f} km")
        print(f"      Średnia długość: {np.mean(line_lengths):.2f} km")
        print(f"      Mediana długości: {np.median(line_lengths):.2f} km")
        
        # Pokaż TOP 5 najdłuższych i najkrótszych
        sorted_details = sorted(line_details, key=lambda x: x['length_km'])
        
        print(f"\n🏆 TOP 5 NAJKRÓTSZYCH LINII:")
        for detail in sorted_details[:5]:
            print(f"      {detail['length_km']:.2f}km - punktów: {detail['coords_count']}")
            
        print(f"\n🏆 TOP 5 NAJDŁUŻSZYCH LINII:")
        for detail in sorted_details[-5:]:
            print(f"      {detail['length_km']:.2f}km - punktów: {detail['coords_count']}")
    
    # ANALIZA 3: Przystanki tramwajowe
    print(f"\n🚏 ANALIZA PRZYSTANKÓW TRAMWAJOWYCH:")
    
    # Sprawdź czy przystanki mają informacje o liniach
    print(f"   Przykładowe przystanki:")
    for i, row in stops_df.head().iterrows():
        stop_info = []
        for col in stops_df.columns:
            if col != 'geometry':
                value = row[col] if pd.notna(row[col]) else 'N/A'
                stop_info.append(f"{col}: {value}")
        print(f"      Przystanek {i}: {', '.join(stop_info)}")
    
    # ANALIZA 4: Odległości między punktami w liniach
    print(f"\n📐 ANALIZA ODLEGŁOŚCI MIĘDZY PUNKTAMI W LINIACH:")
    
    all_distances = []
    line_distances_details = []
    
    for i, row in lines_projected.iterrows():
        if isinstance(row.geometry, LineString):
            coords = list(row.geometry.coords)
            distances = []
            
            for j in range(len(coords) - 1):
                p1 = Point(coords[j])
                p2 = Point(coords[j + 1])
                dist_m = p1.distance(p2)
                distances.append(dist_m)
                all_distances.append(dist_m)
            
            if distances:
                line_dist_detail = {
                    'line_index': i,
                    'segments': len(distances),
                    'min_distance': min(distances),
                    'max_distance': max(distances),
                    'avg_distance': np.mean(distances),
                    'total_length': sum(distances)
                }
                line_distances_details.append(line_dist_detail)
    
    if all_distances:
        print(f"   📊 Statystyki odległości między punktami:")
        print(f"      Minimalna odległość: {min(all_distances):.1f} m")
        print(f"      Maksymalna odległość: {max(all_distances):.1f} m")
        print(f"      Średnia odległość: {np.mean(all_distances):.1f} m")
        print(f"      Mediana odległości: {np.median(all_distances):.1f} m")
        
        # Histogramy odległości
        distances_array = np.array(all_distances)
        print(f"\n📈 Rozkład odległości:")
        print(f"      0-100m: {len(distances_array[distances_array <= 100])} ({len(distances_array[distances_array <= 100])/len(distances_array)*100:.1f}%)")
        print(f"      100-300m: {len(distances_array[(distances_array > 100) & (distances_array <= 300)])} ({len(distances_array[(distances_array > 100) & (distances_array <= 300)])/len(distances_array)*100:.1f}%)")
        print(f"      300-500m: {len(distances_array[(distances_array > 300) & (distances_array <= 500)])} ({len(distances_array[(distances_array > 300) & (distances_array <= 500)])/len(distances_array)*100:.1f}%)")
        print(f"      500-1000m: {len(distances_array[(distances_array > 500) & (distances_array <= 1000)])} ({len(distances_array[(distances_array > 500) & (distances_array <= 1000)])/len(distances_array)*100:.1f}%)")
        print(f"      >1000m: {len(distances_array[distances_array > 1000])} ({len(distances_array[distances_array > 1000])/len(distances_array)*100:.1f}%)")
    
    # ANALIZA 5: Gęstość przystanków
    print(f"\n🗺️ ANALIZA GĘSTOŚCI PRZYSTANKÓW:")
    
    # Konwertuj przystanki do EPSG:2180
    stops_projected = stops_df.to_crs(epsg=2180)
    
    # Oblicz odległości między wszystkimi przystankami
    stop_distances = []
    for i, stop1 in stops_projected.iterrows():
        for j, stop2 in stops_projected.iterrows():
            if i < j:  # Unikaj duplikatów
                dist = stop1.geometry.distance(stop2.geometry)
                stop_distances.append(dist)
    
    if stop_distances:
        print(f"   📊 Odległości między przystankami:")
        print(f"      Minimalna odległość: {min(stop_distances):.1f} m")
        print(f"      Średnia odległość: {np.mean(stop_distances):.1f} m")
        print(f"      Mediana odległości: {np.median(stop_distances):.1f} m")
        
        # Ile przystanków jest bardzo blisko siebie
        close_stops = len([d for d in stop_distances if d < 100])
        print(f"      Przystanki < 100m od siebie: {close_stops}")
        
        very_close_stops = len([d for d in stop_distances if d < 50])
        print(f"      Przystanki < 50m od siebie: {very_close_stops}")
    
    # REKOMENDACJE dla optymalizatora
    print(f"\n💡 REKOMENDACJE DLA OPTYMALIZATORA:")
    print(f"="*50)
    
    if line_lengths:
        avg_length = np.mean(line_lengths)
        print(f"✅ Długość tras: {min(line_lengths):.1f}-{max(line_lengths):.1f}km (śr. {avg_length:.1f}km)")
        print(f"   → Sugerowane ograniczenia: min_total_length={min(line_lengths)*800:.0f}m, max_total_length={max(line_lengths)*1200:.0f}m")
    
    if all_distances:
        reasonable_min = np.percentile(all_distances, 25)  # 25th percentile
        reasonable_max = np.percentile(all_distances, 75)  # 75th percentile
        print(f"✅ Odległości między punktami: {min(all_distances):.0f}-{max(all_distances):.0f}m (rekomendowane: {reasonable_min:.0f}-{reasonable_max:.0f}m)")
        print(f"   → Sugerowane ograniczenia: min_distance_between_stops={reasonable_min:.0f}m, max_distance_between_stops={reasonable_max:.0f}m")
    
    if line_distances_details:
        segments_counts = [detail['segments'] for detail in line_distances_details]
        avg_segments = np.mean(segments_counts)
        print(f"✅ Liczba segmentów w liniach: {min(segments_counts)}-{max(segments_counts)} (śr. {avg_segments:.1f})")
        print(f"   → Sugerowane ograniczenia: min_route_length={min(segments_counts)}, max_route_length={max(segments_counts)}")
    
    print(f"\n🎯 OPTYMALNE PARAMETRY dla RouteConstraints:")
    print(f"   RouteConstraints(")
    if all_distances:
        print(f"       min_distance_between_stops={int(np.percentile(all_distances, 25))},")
        print(f"       max_distance_between_stops={int(np.percentile(all_distances, 75))},")
    if line_lengths:
        print(f"       min_total_length={int(min(line_lengths)*800)},")
        print(f"       max_total_length={int(max(line_lengths)*1200)},")
    if line_distances_details:
        print(f"       min_route_length={min(segments_counts)},")
        print(f"       max_route_length={max(segments_counts)},")
    print(f"       max_angle=45,")
    print(f"       min_distance_from_buildings=3")
    print(f"   )")
    
    return {
        'line_lengths': line_lengths,
        'all_distances': all_distances,
        'line_details': line_details,
        'stops_count': len(stops_df),
        'lines_count': len(lines_df)
    }

if __name__ == "__main__":
    results = analyze_tram_data() 