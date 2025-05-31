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
        
        # Sprawdź czy budynki są w odpowiednim układzie współrzędnych
        if self.buildings_df.crs is None:
            logger.warning("Budynki nie mają określonego CRS - przyjmuję EPSG:4326")
            self.buildings_df.crs = "EPSG:4326"
        
        # Konwertuj budynki do EPSG:2180 (metrowy układ dla Polski)
        if self.buildings_df.crs != "EPSG:2180":
            self.buildings_df_projected = self.buildings_df.to_crs("EPSG:2180")
        else:
            self.buildings_df_projected = self.buildings_df.copy()
            
        logger.info(f"Kalkulator gęstości zainicjalizowany z {len(self.buildings_df)} budynkami")
        
    def calculate_density_at_point(self, lat: float, lon: float, radius: float = None) -> float:
        """
        Oblicza gęstość zabudowy w danym punkcie.
        
        Args:
            lat (float): Latitude (szerokość geograficzna)
            lon (float): Longitude (długość geograficzna)
            radius (float): Promień w metrach (opcjonalny, domyślnie użyje self.radius_meters)
            
        Returns:
            float: Gęstość zabudowy w promieniu od punktu (0-1)
        """
        if radius is None:
            radius = self.radius_meters
            
        try:
            # Utwórz punkt w WGS84
            point_wgs84 = Point(lon, lat)
            point_gdf = gpd.GeoDataFrame(geometry=[point_wgs84], crs="EPSG:4326")
            
            # Konwertuj do EPSG:2180 (metrowy)
            point_projected = point_gdf.to_crs("EPSG:2180").geometry[0]
            
            # Utwórz bufor w metrach
            circle_projected = point_projected.buffer(radius)
            
            # Znajdź budynki w zasięgu
            buildings_in_range = self.buildings_df_projected[
                self.buildings_df_projected.geometry.intersects(circle_projected)
            ]
            
            if buildings_in_range.empty:
                return 0.0
                
            # Oblicz powierzchnię zabudowy w promieniu
            total_building_area = 0.0
            for geom in buildings_in_range.geometry:
                intersection = geom.intersection(circle_projected)
                if not intersection.is_empty:
                    total_building_area += intersection.area
            
            # Oblicz powierzchnię okręgu
            circle_area = circle_projected.area
            
            # Oblicz gęstość zabudowy (0-1)
            if circle_area > 0:
                density = total_building_area / circle_area
                return min(1.0, density)  # Ograniczamy do maksymalnie 1.0
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Błąd obliczania gęstości dla punktu ({lat}, {lon}): {e}")
            return 0.0
    
    def calculate_density_for_route(self, route_points: List[Tuple[float, float]]) -> float:
        """
        Oblicza średnią gęstość zabudowy dla całej trasy.
        
        Args:
            route_points (List[Tuple[float, float]]): Lista punktów trasy (lat, lon)
            
        Returns:
            float: Średnia gęstość zabudowy dla trasy
        """
        if not route_points:
            return 0.0
            
        densities = []
        for lat, lon in route_points:
            density = self.calculate_density_at_point(lat, lon)
            densities.append(density)
            
        return np.mean(densities) if densities else 0.0
    
    def find_high_density_areas(self, threshold: float = 0.1, 
                               grid_size: float = 0.001) -> List[Tuple[float, float, float]]:
        """
        Znajduje obszary o wysokiej gęstości zabudowy.
        
        Args:
            threshold (float): Próg gęstości (0-1)
            grid_size (float): Rozmiar siatki w stopniach
            
        Returns:
            List[Tuple[float, float, float]]: Lista (lat, lon, density)
        """
        high_density_areas = []
        
        # Pobierz granice budynków
        bounds = self.buildings_df.total_bounds
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Utwórz siatkę punktów
        lons = np.arange(min_lon, max_lon, grid_size)
        lats = np.arange(min_lat, max_lat, grid_size)
        
        for lat in lats:
            for lon in lons:
                density = self.calculate_density_at_point(lat, lon)
                if density > threshold:
                    high_density_areas.append((lat, lon, density))
                    
        return sorted(high_density_areas, key=lambda x: x[2], reverse=True)
    
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
                density_map[i, j] = self.calculate_density_at_point(lat, lon)
                
        return density_map, (min_lon, min_lat, max_lon, max_lat) 