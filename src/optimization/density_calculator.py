import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DensityCalculator:
    def __init__(self, buildings_df: gpd.GeoDataFrame, radius_meters: float = 300):
        """
        Inicjalizacja kalkulatora gęstości zabudowy.
        
        Args:
            buildings_df (gpd.GeoDataFrame): DataFrame zawierający dane o budynkach
            radius_meters (float): Promień w metrach, w którym obliczana jest gęstość
        """
        self.buildings_df = buildings_df
        self.radius_meters = radius_meters
        
        # Konwersja promienia z metrów na stopnie (przybliżone)
        self.radius_degrees = radius_meters / 111000  # 1 stopień ≈ 111km
        
    def calculate_density_at_point(self, point: Tuple[float, float]) -> float:
        """
        Oblicza gęstość zabudowy w danym punkcie.
        
        Args:
            point (Tuple[float, float]): Współrzędne punktu (latitude, longitude)
            
        Returns:
            float: Gęstość zabudowy w promieniu radius_meters od punktu
        """
        lat, lon = point
        
        # Tworzenie okręgu wokół punktu
        circle = Point(lon, lat).buffer(self.radius_degrees)
        
        # Znajdowanie budynków w zasięgu
        buildings_in_range = self.buildings_df[self.buildings_df.geometry.intersects(circle)]
        
        if buildings_in_range.empty:
            return 0.0
            
        # Obliczanie powierzchni zabudowy w promieniu
        total_building_area = buildings_in_range.geometry.area.sum()
        
        # Obliczanie powierzchni okręgu
        circle_area = circle.area
        
        # Obliczanie gęstości zabudowy
        density = total_building_area / circle_area
        
        return density
    
    def calculate_density_for_route(self, route_points: List[Tuple[float, float]]) -> float:
        """
        Oblicza średnią gęstość zabudowy dla całej trasy.
        
        Args:
            route_points (List[Tuple[float, float]]): Lista punktów trasy
            
        Returns:
            float: Średnia gęstość zabudowy dla trasy
        """
        densities = [self.calculate_density_at_point(point) for point in route_points]
        return np.mean(densities)
    
    def get_density_map(self, grid_size: int = 100) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Tworzy mapę gęstości zabudowy dla całego obszaru.
        
        Args:
            grid_size (int): Liczba punktów w siatce w każdym wymiarze
            
        Returns:
            Tuple[np.ndarray, Tuple[float, float, float, float]]: 
                - Mapa gęstości zabudowy
                - Granice obszaru (min_lon, min_lat, max_lon, max_lat)
        """
        bounds = self.buildings_df.total_bounds
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Tworzenie siatki punktów
        lons = np.linspace(min_lon, max_lon, grid_size)
        lats = np.linspace(min_lat, max_lat, grid_size)
        
        density_map = np.zeros((grid_size, grid_size))
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                density_map[i, j] = self.calculate_density_at_point((lat, lon))
                
        return density_map, bounds 