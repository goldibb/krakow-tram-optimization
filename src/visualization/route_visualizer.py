import folium
from folium import plugins
import numpy as np
import geopandas as gpd
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import branca.colormap as cm
import logging

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
        self.center_lat = buildings_df.geometry.centroid.y.mean()
        self.center_lon = buildings_df.geometry.centroid.x.mean()
        
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