import folium
from folium import plugins
import numpy as np
import geopandas as gpd
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import branca.colormap as cm
import logging
import time
from IPython.display import display, clear_output
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RouteVisualizer:
    def __init__(self, buildings_df: gpd.GeoDataFrame, streets_df: gpd.GeoDataFrame):
        """
        Inicjalizacja wizualizatora tras.
        
        Args:
            buildings_df: DataFrame z budynkami
            streets_df: DataFrame z ulicami
        """
        self.buildings_df = buildings_df
        self.streets_df = streets_df
        
        # Obliczenie centrum na podstawie granic budynków w WGS84
        bounds = buildings_df.total_bounds
        min_lon, min_lat, max_lon, max_lat = bounds
        
        self.center_lat = (min_lat + max_lat) / 2
        self.center_lon = (min_lon + max_lon) / 2
        
    def create_base_map(self, zoom_start: int = 13) -> folium.Map:
        """
        Tworzy podstawową mapę z budynkami i ulicami.
        
        Args:
            zoom_start: Początkowy poziom przybliżenia mapy
            
        Returns:
            folium.Map: Obiekt mapy
        """
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Dodawanie warstwy budynków
        buildings_style = {
            'fillColor': '#808080',
            'color': '#000000',
            'weight': 1,
            'fillOpacity': 0.3
        }
        
        folium.GeoJson(
            self.buildings_df,
            style_function=lambda x: buildings_style,
            name='Budynki'
        ).add_to(m)
        
        # Dodawanie warstwy ulic
        streets_style = {
            'color': '#404040',
            'weight': 2,
            'opacity': 0.7
        }
        
        folium.GeoJson(
            self.streets_df,
            style_function=lambda x: streets_style,
            name='Ulice'
        ).add_to(m)
        
        # Dodawanie kontrolki warstw
        folium.LayerControl().add_to(m)
        
        return m
    
    def plot_route(
        self,
        route: List[Tuple[float, float]],
        map_obj: Optional[folium.Map] = None,
        route_name: str = "Optymalna trasa",
        color: str = 'blue',
        weight: int = 4,
        opacity: float = 0.8
    ) -> folium.Map:
        """
        Dodaje trasę do mapy.
        
        Args:
            route: Lista punktów trasy
            map_obj: Obiekt mapy (opcjonalny)
            route_name: Nazwa trasy
            color: Kolor linii
            weight: Grubość linii
            opacity: Przezroczystość linii
            
        Returns:
            folium.Map: Obiekt mapy z dodaną trasą
        """
        if map_obj is None:
            map_obj = self.create_base_map()
            
        # Dodawanie linii trasy
        folium.PolyLine(
            locations=[[lat, lon] for lat, lon in route],
            color=color,
            weight=weight,
            opacity=opacity,
            name=route_name
        ).add_to(map_obj)
        
        # Dodawanie markerów przystanków
        for i, (lat, lon) in enumerate(route):
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color='red',
                fill=True,
                popup=f'Przystanek {i+1}',
                name=f'Przystanek {i+1}'
            ).add_to(map_obj)
            
        return map_obj
    
    def plot_density_map(
        self,
        density_map: np.ndarray,
        bounds: Tuple[float, float, float, float],
        map_obj: Optional[folium.Map] = None,
        opacity: float = 0.6
    ) -> folium.Map:
        """
        Dodaje mapę gęstości zabudowy do mapy.
        
        Args:
            density_map: Mapa gęstości zabudowy
            bounds: Granice obszaru (min_lon, min_lat, max_lon, max_lat)
            map_obj: Obiekt mapy (opcjonalny)
            opacity: Przezroczystość warstwy
            
        Returns:
            folium.Map: Obiekt mapy z dodaną mapą gęstości
        """
        if map_obj is None:
            map_obj = self.create_base_map()
            
        # Tworzenie skali kolorów
        colormap = cm.LinearColormap(
            ['green', 'yellow', 'red'],
            vmin=0,
            vmax=density_map.max(),
            caption='Gęstość zabudowy'
        )
        
        # Dodawanie warstwy gęstości
        folium.raster_layers.ImageOverlay(
            density_map,
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            opacity=opacity,
            colormap=colormap,
            name='Gęstość zabudowy'
        ).add_to(map_obj)
        
        colormap.add_to(map_obj)
        
        return map_obj
    
    def plot_optimization_results(
        self,
        best_route: List[Tuple[float, float]],
        density_map: np.ndarray,
        bounds: Tuple[float, float, float, float],
        score: float,
        save_path: Optional[str] = None
    ) -> folium.Map:
        """
        Tworzy kompletną wizualizację wyników optymalizacji.
        
        Args:
            best_route: Najlepsza znaleziona trasa
            density_map: Mapa gęstości zabudowy
            bounds: Granice obszaru
            score: Ocena trasy
            save_path: Ścieżka do zapisania mapy (opcjonalna)
            
        Returns:
            folium.Map: Obiekt mapy z wynikami
        """
        m = self.create_base_map()
        
        # Dodawanie mapy gęstości
        m = self.plot_density_map(density_map, bounds, m)
        
        # Dodawanie trasy
        m = self.plot_route(best_route, m)
        
        # Dodawanie informacji o ocenie
        folium.Popup(
            f'Ocena trasy: {score:.2f}',
            max_width=300
        ).add_to(m)
        
        if save_path:
            m.save(save_path)
            logger.info(f"Mapa zapisana w: {save_path}")
            
        return m
    
    def plot_optimization_progress(
        self,
        scores_history: List[float],
        save_path: Optional[str] = None
    ) -> None:
        """
        Tworzy wykres postępu optymalizacji.
        
        Args:
            scores_history: Historia najlepszych wyników
            save_path: Ścieżka do zapisania wykresu (opcjonalna)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(scores_history, 'b-', label='Najlepszy wynik')
        plt.xlabel('Pokolenie')
        plt.ylabel('Ocena trasy')
        plt.title('Postęp optymalizacji')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Wykres zapisany w: {save_path}")
            
        plt.show()
        
    def visualize_optimization_process(
        self,
        population: List[List[Tuple[float, float]]],
        scores: List[float],
        generation: int,
        best_route: List[Tuple[float, float]],
        best_score: float,
        bounds: Tuple[float, float, float, float],
        density_map: np.ndarray,
        update_interval: float = 0.5
    ) -> None:
        """
        Wizualizuje proces optymalizacji w czasie rzeczywistym.
        
        Args:
            population: Aktualna populacja tras
            scores: Oceny tras w populacji
            generation: Numer aktualnego pokolenia
            best_route: Najlepsza znaleziona trasa
            best_score: Ocena najlepszej trasy
            bounds: Granice obszaru
            density_map: Mapa gęstości zabudowy
            update_interval: Czas między aktualizacjami wizualizacji (w sekundach)
        """
        m = self.create_base_map()
        
        # Dodawanie mapy gęstości
        m = self.plot_density_map(density_map, bounds, m)
        
        # Dodawanie wszystkich tras z populacji (przezroczyste)
        for i, route in enumerate(population):
            self.plot_route(
                route, m,
                route_name=f"Trasa {i+1}",
                color='gray',
                opacity=0.2
            )
        
        # Dodawanie najlepszej trasy (wyraźna)
        self.plot_route(
            best_route, m,
            route_name="Najlepsza trasa",
            color='blue',
            opacity=0.8
        )
        
        # Dodawanie informacji o pokoleniu i ocenie
        folium.Popup(
            f'Pokolenie: {generation}\n'
            f'Najlepsza ocena: {best_score:.2f}\n'
            f'Średnia ocena: {np.mean(scores):.2f}',
            max_width=300
        ).add_to(m)
        
        # Wyświetlanie mapy
        display(m)
        clear_output(wait=True)
        time.sleep(update_interval)
    
    def plot_stops_only(
        self,
        route: List[Tuple[float, float]],
        map_obj: Optional[folium.Map] = None,
        route_name: str = "Przystanki",
        color: str = 'blue',
        show_numbers: bool = True
    ) -> folium.Map:
        """
        Dodaje tylko przystanki do mapy bez łączących linii (unika "przeskakiwania").
        
        Args:
            route: Lista punktów przystanków
            map_obj: Obiekt mapy (opcjonalny)
            route_name: Nazwa tras
            color: Kolor markerów
            show_numbers: Czy pokazywać numery przystanków
            
        Returns:
            folium.Map: Obiekt mapy z dodanymi przystankami
        """
        if map_obj is None:
            map_obj = self.create_base_map()
            
        # Dodawanie markerów przystanków BEZ łączącej linii
        for i, (lat, lon) in enumerate(route):
            popup_text = f'{route_name} - Przystanek {i+1}' if show_numbers else f'{route_name}'
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=popup_text,
                name=popup_text
            ).add_to(map_obj)
            
            # Opcjonalnie dodaj numer przystanku jako tekst
            if show_numbers:
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 12px; color: white; font-weight: bold; text-align: center; background-color: {color}; border-radius: 50%; width: 20px; height: 20px; line-height: 20px;">{i+1}</div>',
                        icon_size=(20, 20),
                        icon_anchor=(10, 10)
                    )
                ).add_to(map_obj)
            
        return map_obj
    
    def plot_route_segments(
        self,
        route: List[Tuple[float, float]],
        map_obj: Optional[folium.Map] = None,
        route_name: str = "Trasa segmentowa",
        color: str = 'blue',
        weight: int = 4,
        opacity: float = 0.8,
        max_segment_length: float = 2000  # maksymalna długość segmentu w metrach
    ) -> folium.Map:
        """
        Dodaje trasę do mapy jako oddzielne segmenty (unika długich "przeskoków").
        
        Args:
            route: Lista punktów trasy
            map_obj: Obiekt mapy (opcjonalny)
            route_name: Nazwa trasy
            color: Kolor linii
            weight: Grubość linii
            opacity: Przezroczystość linii
            max_segment_length: Maksymalna długość segmentu w metrach
            
        Returns:
            folium.Map: Obiekt mapy z dodaną trasą
        """
        if map_obj is None:
            map_obj = self.create_base_map()
            
        # Funkcja pomocnicza do obliczania odległości w metrach
        def calculate_distance_meters(lat1, lon1, lat2, lon2):
            R = 6371000  # promień Ziemi w metrach
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            return 2 * R * math.asin(math.sqrt(a))
            
        # Dodawanie segmentów trasy (tylko te o rozsądnej długości)
        for i in range(len(route) - 1):
            lat1, lon1 = route[i]
            lat2, lon2 = route[i + 1]
            
            # Sprawdź długość segmentu
            segment_length = calculate_distance_meters(lat1, lon1, lat2, lon2)
            
            if segment_length <= max_segment_length:
                # Dodaj segment jako osobną linię
                folium.PolyLine(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    color=color,
                    weight=weight,
                    opacity=opacity,
                    name=f"{route_name} - Segment {i+1}",
                    popup=f"Segment {i+1}-{i+2}: {segment_length:.0f}m"
                ).add_to(map_obj)
            else:
                # Segment za długi - prawdopodobnie "przeskok", rysuj tylko przystanki
                folium.CircleMarker(
                    location=[lat1, lon1],
                    radius=6,
                    color='orange',
                    fill=True,
                    popup=f'Przystanek {i+1} (przeskok: {segment_length:.0f}m)',
                    name=f'Przystanek {i+1}'
                ).add_to(map_obj)
                
                if i == len(route) - 2:  # Ostatni punkt
                    folium.CircleMarker(
                        location=[lat2, lon2],
                        radius=6,
                        color='orange',
                        fill=True,
                        popup=f'Przystanek {i+2}',
                        name=f'Przystanek {i+2}'
                    ).add_to(map_obj)
        
        # Dodawanie markerów przystanków
        for i, (lat, lon) in enumerate(route):
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color='red',
                fill=True,
                popup=f'Przystanek {i+1}',
                name=f'Przystanek {i+1}'
            ).add_to(map_obj)
            
        return map_obj
    
    def plot_route_sequential(self, route: List[Tuple[float, float]], map_obj, 
                            route_name: str = "Sequential Route", color: str = 'green', 
                            max_segment_length: float = 800):
        """
        NOWA METODA: Wizualizuje trasę sekwencyjną bez skoków z kontrolą odległości.
        
        Args:
            route: Lista punktów trasy (lat, lon)
            map_obj: Obiekt mapy Folium
            route_name: Nazwa trasy
            color: Kolor linii
            max_segment_length: Maksymalna długość segmentu w metrach (długie segmenty = skoki)
        """
        if not route or len(route) < 2:
            return
        
        logger.info(f"🎨 Wizualizacja sekwencyjna: {len(route)} punktów, max segment {max_segment_length}m")
        
        # Dodaj przystanki jako markery
        for i, (lat, lon) in enumerate(route):
            # Różne ikony dla pierwszego, ostatniego i pośrednich przystanków
            if i == 0:
                icon_color = 'green'
                icon = 'play'
                popup_text = f"🚀 START\nPunkt {i+1}\nLat: {lat:.6f}, Lon: {lon:.6f}"
            elif i == len(route) - 1:
                icon_color = 'red'
                icon = 'stop'
                popup_text = f"🏁 KONIEC\nPunkt {i+1}\nLat: {lat:.6f}, Lon: {lon:.6f}"
            else:
                icon_color = 'blue'
                icon = 'record'
                popup_text = f"🚏 PRZYSTANEK\nPunkt {i+1}\nLat: {lat:.6f}, Lon: {lon:.6f}"
            
            # Dodaj marker
            folium.Marker(
                [lat, lon],
                popup=popup_text,
                tooltip=f"Punkt {i+1}",
                icon=folium.Icon(color=icon_color, icon=icon)
            ).add_to(map_obj)
        
        # Dodaj linie między przystankami z kontrolą skoków
        connected_segments = 0
        skipped_segments = 0
        
        for i in range(len(route) - 1):
            point1 = route[i]
            point2 = route[i + 1]
            
            # Oblicz odległość między punktami (przybliżona)
            lat1, lon1 = point1
            lat2, lon2 = point2
            
            # Prosta formuła haversine dla odległości (w metrach)
            import math
            R = 6371000  # Promień Ziemi w metrach
            
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat/2) * math.sin(dlat/2) + 
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                 math.sin(dlon/2) * math.sin(dlon/2))
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c
            
            if distance <= max_segment_length:
                # Normalny segment - narysuj zieloną linię
                folium.PolyLine(
                    [point1, point2],
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=f"Segment {i+1}-{i+2}: {distance:.0f}m"
                ).add_to(map_obj)
                connected_segments += 1
                
            else:
                # Skok wykryty - narysuj czerwoną przerywaną linię
                folium.PolyLine(
                    [point1, point2],
                    color='red',
                    weight=2,
                    opacity=0.5,
                    dash_array='10, 10',
                    popup=f"⚠️ SKOK {i+1}-{i+2}: {distance:.0f}m > {max_segment_length}m"
                ).add_to(map_obj)
                skipped_segments += 1
                
                # Dodaj marker ostrzeżenia w środku skoku
                mid_lat = (lat1 + lat2) / 2
                mid_lon = (lon1 + lon2) / 2
                
                folium.Marker(
                    [mid_lat, mid_lon],
                    popup=f"⚠️ SKOK: {distance:.0f}m",
                    tooltip="Wykryto skok!",
                    icon=folium.Icon(color='orange', icon='warning-sign')
                ).add_to(map_obj)
        
        # Dodaj legendę z informacjami o trasie
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 300px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>{route_name}</h4>
        <p><b>📍 Punkty:</b> {len(route)}</p>
        <p><b>✅ Połączenia:</b> {connected_segments}</p>
        <p><b>⚠️ Skoki:</b> {skipped_segments}</p>
        <p><b>🎯 Max segment:</b> {max_segment_length}m</p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))
        
        logger.info(f"   ✅ {connected_segments} normalnych połączeń")
        logger.info(f"   ⚠️ {skipped_segments} skoków wykrytych")
        
        # Automatyczne dopasowanie widoku mapy
        if route:
            lats = [lat for lat, lon in route]
            lons = [lon for lat, lon in route]
            
            sw = [min(lats), min(lons)]
            ne = [max(lats), max(lons)]
            map_obj.fit_bounds([sw, ne], padding=(20, 20)) 