import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import Tuple, List, Optional
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
        
    def calculate_density_at_point(self, lon: float, lat: float, radius: float = None) -> float:
        """
        Oblicza gęstość zabudowy w danym punkcie.
        
        Args:
            lon (float): Longitude
            lat (float): Latitude  
            radius (float): Promień w metrach (opcjonalny, domyślnie użyje self.radius_meters)
            
        Returns:
            float: Gęstość zabudowy w promieniu od punktu
        """
        if radius is None:
            radius = self.radius_meters
            
        # Konwersja promienia z metrów na stopnie (przybliżone)
        radius_degrees = radius / 111000  # 1 stopień ≈ 111km
        
        # Tworzenie okręgu wokół punktu
        circle = Point(lon, lat).buffer(radius_degrees)
        
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
        densities = [self.calculate_density_at_point(point[0], point[1]) for point in route_points]
        return np.mean(densities)
    
    def get_density_map(self, grid_size: float = 0.001, bounds: Optional[Tuple[float, float, float, float]] = None) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Tworzy mapę gęstości zabudowy dla całego obszaru lub podanego obszaru.
        
        Args:
            grid_size (float): Rozmiar siatki w stopniach
            bounds (Optional[Tuple[float, float, float, float]]): Granice obszaru (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            Tuple[np.ndarray, Tuple[float, float, float, float]]: 
                - Mapa gęstości zabudowy
                - Granice obszaru (min_lon, min_lat, max_lon, max_lat)
        """
        if bounds is not None:
            min_lon, min_lat, max_lon, max_lat = bounds
        else:
            total_bounds = self.buildings_df.total_bounds
            min_lon, min_lat, max_lon, max_lat = total_bounds
        
        # Tworzenie siatki punktów
        lons = np.arange(min_lon, max_lon, grid_size)
        lats = np.arange(min_lat, max_lat, grid_size)
        
        density_map = np.zeros((len(lats), len(lons)))
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                density_map[i, j] = self.calculate_density_at_point(lon, lat)
                
        return density_map, (min_lon, min_lat, max_lon, max_lat) 