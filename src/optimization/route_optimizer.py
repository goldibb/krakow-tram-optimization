import numpy as np
from typing import List, Tuple, Dict, Optional
import random
from dataclasses import dataclass
import logging
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
import networkx as nx
from shapely.ops import unary_union
from scipy.spatial import distance
from scipy.spatial import cKDTree
import time
from .density_calculator import DensityCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RouteConstraints:
    min_distance_between_stops: float = 200  # minimalna odległość między przystankami w metrach
    max_distance_between_stops: float = 1500  # maksymalna odległość między przystankami w metrach
    max_angle: float = 60  # maksymalny kąt zakrętu w stopniach
    min_route_length: int = 3  # minimalna liczba przystanków
    max_route_length: int = 20  # maksymalna liczba przystanków
    min_total_length: float = 1000  # minimalna długość całkowita trasy w metrach
    max_total_length: float = 15000  # maksymalna długość całkowita trasy w metrach
    min_distance_from_buildings: float = 3  # minimalna odległość od budynków w metrach
    angle_weight: float = 0.1  # waga dla kryterium minimalizacji kątów

class RouteOptimizer:
    def _prepare_existing_lines(self) -> List[LineString]:
        """
        Przygotowuje geometrie istniejących linii tramwajowych.
        
        Returns:
            List[LineString]: Lista geometrii istniejących linii
        """
        existing_lines = []
        if self.lines_df is not None:
            for _, row in self.lines_df.iterrows():
                if isinstance(row.geometry, LineString):
                    existing_lines.append(row.geometry)
        return existing_lines

    def _create_buildings_buffer(self) -> Polygon:
        """
        Tworzy bufor wokół budynków.
        
        Returns:
            Polygon: Bufor wokół budynków
        """
        buildings_union = unary_union(self.buildings_projected.geometry)
        return buildings_union.buffer(self.constraints.min_distance_from_buildings)

    def __init__(
        self,
        buildings_df: gpd.GeoDataFrame,
        streets_df: gpd.GeoDataFrame,
        stops_df: Optional[gpd.GeoDataFrame] = None,
        lines_df: Optional[gpd.GeoDataFrame] = None,
        constraints: Optional[RouteConstraints] = None,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        population_weight: float = 0.7,  # waga dla kryterium gęstości zaludnienia
        distance_weight: float = 0.2,    # waga dla kryterium odległości
        angle_weight: float = 0.1,       # waga dla kryterium minimalizacji kątów
    ):
        """
        Inicjalizacja optymalizatora tras.
        
        Args:
            buildings_df: DataFrame z budynkami
            streets_df: DataFrame z ulicami
            stops_df: DataFrame z istniejącymi przystankami (opcjonalne)
            lines_df: DataFrame z istniejącymi liniami tramwajowymi (opcjonalne)
            constraints: Ograniczenia dla trasy (opcjonalne)
            population_size: Rozmiar populacji
            generations: Liczba pokoleń
            mutation_rate: Współczynnik mutacji
            crossover_rate: Współczynnik krzyżowania
            population_weight: Waga dla kryterium gęstości zaludnienia
            distance_weight: Waga dla kryterium odległości
            angle_weight: Waga dla kryterium minimalizacji kątów
        """
        self.buildings_df = buildings_df
        self.streets_df = streets_df
        self.stops_df = stops_df
        self.lines_df = lines_df
        self.constraints = constraints or RouteConstraints()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Normalizacja wag - muszą się sumować do 1
        total_weight = population_weight + distance_weight + angle_weight
        self.population_weight = population_weight / total_weight
        self.distance_weight = distance_weight / total_weight  
        self.angle_weight = angle_weight / total_weight
        
        # Transformacja do układu współrzędnych rzutowanych (EPSG:2180 dla Polski)
        self.buildings_projected = buildings_df.to_crs(epsg=2180)
        self.streets_projected = streets_df.to_crs(epsg=2180)
        if stops_df is not None:
            self.stops_projected = stops_df.to_crs(epsg=2180)
        if lines_df is not None:
            self.lines_projected = lines_df.to_crs(epsg=2180)
        
        # Tworzenie grafu sieci ulic
        logger.info("Tworzenie grafu sieci ulic...")
        self.street_graph = self._create_street_graph()
        logger.info(f"Graf utworzony z {self.street_graph.number_of_nodes()} węzłami i {self.street_graph.number_of_edges()} krawędziami")
        
        # OPTYMALIZACJA: Tworzenie spatial index dla szybkiego wyszukiwania
        logger.info("Tworzenie spatial index...")
        self._create_spatial_index()
        
        # Cache dla najbliższych punktów
        self._nearest_point_cache = {}
        
        # Przygotowanie istniejących linii i buforów
        self.existing_lines = self._prepare_existing_lines() if lines_df is not None else []
        self.buildings_buffer = self._create_buildings_buffer()
        
        # Inicjalizacja kalkulatora gęstości
        logger.info("Inicjalizacja kalkulatora gęstości...")
        self.density_calculator = DensityCalculator(self.buildings_df, radius_meters=300)
        
        # Set do śledzenia używanych przystanków w całym systemie
        self.used_stops = set()

    def _create_street_graph(self) -> nx.Graph:
        """
        Tworzy graf sieci ulic na podstawie danych OSM.
        Skupia się tylko na najgęściej zaludnionych obszarach dla maksymalnej efektywności.
        
        Returns:
            nx.Graph: Graf sieci ulic
        """
        G = nx.Graph()
        
        # SZYBKIE WYSZUKIWANIE NAJGĘŚCIEJ ZALUDNIONYCH OBSZARÓW
        logger.info("Wyszukiwanie najgęściej zaludnionych obszarów...")
        
        if self.stops_df is not None and len(self.stops_df) > 0 and len(self.buildings_projected) > 0:
            # 1. Znajdź TOP 5 najgęściej zaludnionych przystanków
            top_density_stops = self._find_top_density_stops(top_n=5)
            logger.info(f"Znaleziono {len(top_density_stops)} przystanków o najwyższej gęstości zaludnienia")
            
            # 2. Utwórz bufor 800m wokół TOP przystanków
            buffer_distance = 800  # 800m - zasięg pieszej dostępności
            relevant_areas = []
            
            for stop_coords in top_density_stops:
                # Konwertuj do EPSG:2180
                stop_gdf = gpd.GeoDataFrame(
                    geometry=[Point(stop_coords[1], stop_coords[0])],  # lon, lat
                    crs="EPSG:4326"
                ).to_crs(epsg=2180)
                
                stop_buffer = stop_gdf.geometry.buffer(buffer_distance)[0]
                relevant_areas.append(stop_buffer)
            
            # 3. Połącz wszystkie obszary
            if relevant_areas:
                relevant_area = unary_union(relevant_areas)
                logger.info("Utworzono bufor 800m wokół najgęstszych przystanków")
            else:
                # Fallback - bufor wokół wszystkich przystanków ale mniejszy
                stops_buffers = self.stops_projected.geometry.buffer(500)
                relevant_area = unary_union(stops_buffers.head(10))  # tylko 10 pierwszych
                logger.info("Fallback: bufor 500m wokół 10 pierwszych przystanków")
            
            # 4. Filtruj ulice do wybranych obszarów
            logger.info("Filtrowanie ulic do wybranych obszarów...")
            streets_in_relevant_area = self.streets_projected[
                self.streets_projected.geometry.intersects(relevant_area)
            ]
            
            logger.info(f"Ograniczono z {len(self.streets_projected)} do {len(streets_in_relevant_area)} ulic")
            
            # 5. Jeśli nadal za dużo, weź próbkę z priorytetem dla głównych dróg
            if len(streets_in_relevant_area) > 2000:
                # Próbka z preferencją dla większych ulic (jeśli mają większą powierzchnię)
                streets_filtered = streets_in_relevant_area.sample(n=2000, random_state=42)
                logger.info(f"Ograniczono do {len(streets_filtered)} ulic (próbka)")
            else:
                streets_filtered = streets_in_relevant_area
                
        else:
            # Ultra-szybki fallback
            logger.warning("Szybki tryb: używam tylko 1500 losowych ulic")
            streets_filtered = self.streets_projected.sample(n=min(1500, len(self.streets_projected)), random_state=42)
        
        logger.info(f"Finalna liczba ulic: {len(streets_filtered)}")
        
        # Dodawanie węzłów (skrzyżowania) - tylko jeśli mamy rozsądną liczbę ulic
        if len(streets_filtered) > 5000:
            logger.warning(f"Nadal za dużo ulic ({len(streets_filtered)}), ograniczam do 1000")
            streets_filtered = streets_filtered.head(1000)
        
        for idx, row in streets_filtered.iterrows():
            coords = list(row.geometry.coords)
            for i in range(len(coords) - 1):
                point1_epsg2180 = coords[i]
                point2_epsg2180 = coords[i + 1]
                
                dist = self._calculate_distance(point1_epsg2180, point2_epsg2180, is_wgs84=False)
                
                if dist > 0:
                    G.add_edge(point1_epsg2180, point2_epsg2180, weight=dist)
        
        return G
    
    def calculate_density_score(self, route: List[Tuple[float, float]], radius: float = 300) -> float:
        """
        Oblicza ocenę trasy na podstawie gęstości zaludnienia wokół przystanków.
        
        Args:
            route: Lista punktów trasy
            radius: Promień w metrach, w którym szukamy budynków
            
        Returns:
            float: Ocena trasy (0-1)
        """
        total_score = 0
        
        # Konwersja punktów trasy do GeoDataFrame
        points = [Point(lon, lat) for lat, lon in route]
        route_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
        
        # Konwersja do układu EPSG:2180 dla obliczeń w metrach
        route_gdf = route_gdf.to_crs(epsg=2180)
        
        for point in route_gdf.geometry:
            # Znajdź wszystkie budynki w zadanym promieniu
            buildings_in_radius = self.buildings_projected[
                self.buildings_projected.geometry.distance(point) <= radius
            ]
            
            # Oblicz gęstość (np. na podstawie powierzchni budynków)
            if not buildings_in_radius.empty:
                density = buildings_in_radius.geometry.area.sum() / (np.pi * radius**2)
                total_score += density
        
        return total_score / len(route) if route else 0
    
    def _validate_coordinates(self, point: Tuple[float, float], is_wgs84: bool = False) -> bool:
        """
        Sprawdza czy współrzędne są prawidłowe.
        
        Args:
            point: Punkt (x, y) w układzie EPSG:2180 lub (lat, lon) w WGS84
            is_wgs84: Czy współrzędne są w układzie WGS84
            
        Returns:
            bool: True jeśli współrzędne są prawidłowe
        """
        try:
            x, y = point
            # Sprawdzenie czy wartości nie są None lub NaN
            if any(p is None or np.isnan(p) for p in (x, y)):
                return False
                
            # Sprawdzenie czy wartości są liczbami
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                return False
            
            if is_wgs84:
                # Sprawdzenie czy wartości są w rozsądnym zakresie dla WGS84 (Kraków i okolice)
                # Rozszerzony zakres dla większego obszaru Krakowa
                if not (49.8 <= x <= 50.3 and 19.5 <= y <= 20.5):
                    logger.debug(f"Współrzędne WGS84 poza zakresem: lat={x} (49.8-50.3), lon={y} (19.5-20.5)")
                    return False
            else:
                # Sprawdzenie czy wartości są w rozsądnym zakresie dla EPSG:2180
                # Bounds for Poland EPSG:2180: northing (x): 125837-908411, easting (y): 144693-876500
                if not (125000 <= x <= 910000 and 140000 <= y <= 880000):
                    logger.debug(f"Współrzędne EPSG:2180 poza zakresem: x={x} (125000-910000), y={y} (140000-880000)")
                    return False
                
            return True
            
        except Exception:
            return False

    def calculate_distance_score(self, route: List[Tuple[float, float]]) -> float:
        """
        Oblicza ocenę trasy na podstawie odległości między przystankami.
        
        Args:
            route: Lista punktów trasy
            
        Returns:
            float: Ocena trasy (0-1)
        """
        if len(route) < 2:
            return 0
            
        total_distance = 0
        for i in range(len(route) - 1):
            # Sprawdź czy współrzędne są prawidłowe (WGS84)
            if not (self._validate_coordinates(route[i], is_wgs84=True) and 
                   self._validate_coordinates(route[i + 1], is_wgs84=True)):
                logger.warning(f"Nieprawidłowe współrzędne w trasie: {route[i]} -> {route[i + 1]}")
                return 0
                
            # Użyj unifiednej metody obliczania odległości
            dist = self._calculate_distance(route[i], route[i + 1], is_wgs84=True)
            
            if dist == 0:  # Błąd podczas obliczania odległości
                return 0
                
            if dist < self.constraints.min_distance_between_stops:
                return 0  # Kara za zbyt małą odległość
            if dist > self.constraints.max_distance_between_stops:
                return 0  # Kara za zbyt dużą odległość
                
            total_distance += dist
            
        # Normalizacja wyniku (im mniejsza odległość, tym lepszy wynik)
        max_possible_distance = self.constraints.max_distance_between_stops * (len(route) - 1)
        return 1 - (total_distance / max_possible_distance)
    
    def calculate_angle_score(self, route: List[Tuple[float, float]]) -> float:
        """
        Oblicza ocenę trasy na podstawie minimalizacji kątów zakrętu.
        
        Args:
            route: Lista punktów trasy
            
        Returns:
            float: Ocena trasy (0-1, wyższe wartości dla prostszych tras)
        """
        if len(route) < 3:
            return 1.0  # Brak zakrętów dla tras z 2 lub mniej punktów
            
        total_angle_penalty = 0
        angle_count = 0
        
        for i in range(1, len(route) - 1):
            # Sprawdź czy współrzędne są prawidłowe
            if not (self._validate_coordinates(route[i-1], is_wgs84=True) and 
                   self._validate_coordinates(route[i], is_wgs84=True) and
                   self._validate_coordinates(route[i+1], is_wgs84=True)):
                continue
                
            angle = self._calculate_angle(route[i-1], route[i], route[i+1])
            
            # Kara za ostre zakręty - im większy kąt, tym większa kara
            # Korzystamy z odchylenia od linii prostej (180°)
            angle_deviation = abs(180 - angle)
            angle_penalty = angle_deviation / 180.0  # normalizacja do 0-1
            
            total_angle_penalty += angle_penalty
            angle_count += 1
            
        if angle_count == 0:
            return 1.0
            
        # Średnia kara za kąty - im mniejsza, tym lepszy wynik
        average_angle_penalty = total_angle_penalty / angle_count
        return 1.0 - average_angle_penalty

    def _find_connecting_path(self, start_point: Tuple[float, float], end_point: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Znajduje ścieżkę łączącą dwa punkty przez sieć ulic używając algorytmu A*.
        
        Args:
            start_point: Punkt początkowy (lat, lon) w WGS84
            end_point: Punkt końcowy (lat, lon) w WGS84
            
        Returns:
            List[Tuple[float, float]]: Lista punktów ścieżki w WGS84 lub [start_point, end_point] jeśli nie znaleziono
        """
        # Znajdź najbliższe węzły w grafie dla obu punktów
        start_node = self._find_nearest_point_in_graph(start_point, max_distance=1000)
        end_node = self._find_nearest_point_in_graph(end_point, max_distance=1000)
        
        if start_node is None or end_node is None:
            logger.warning(f"Nie znaleziono węzłów w grafie dla punktów: {start_point} -> {end_point}")
            return [start_point, end_point]
        
        # Konwertuj węzły do formatu używanego przez graf (EPSG:2180)
        try:
            start_gdf = gpd.GeoDataFrame(
                geometry=[Point(start_node[1], start_node[0])],  # lon, lat
                crs="EPSG:4326"
            ).to_crs(epsg=2180)
            start_epsg2180 = (start_gdf.geometry.x[0], start_gdf.geometry.y[0])
            
            end_gdf = gpd.GeoDataFrame(
                geometry=[Point(end_node[1], end_node[0])],  # lon, lat
                crs="EPSG:4326"
            ).to_crs(epsg=2180)
            end_epsg2180 = (end_gdf.geometry.x[0], end_gdf.geometry.y[0])
            
        except Exception as e:
            logger.warning(f"Błąd konwersji współrzędnych: {str(e)}")
            return [start_point, end_point]
        
        # Sprawdź czy węzły istnieją w grafie
        if start_epsg2180 not in self.street_graph or end_epsg2180 not in self.street_graph:
            logger.warning(f"Węzły nie istnieją w grafie: {start_epsg2180}, {end_epsg2180}")
            return [start_point, end_point]
        
        try:
            # Użyj A* do znajdowania najkrótszej ścieżki
            def heuristic(node1, node2):
                """Funkcja heurystyczna dla A* - odległość euklidesowa"""
                return ((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)**0.5
            
            path = nx.astar_path(
                self.street_graph, 
                start_epsg2180, 
                end_epsg2180, 
                heuristic=heuristic,
                weight='weight'
            )
            
            # Konwertuj ścieżkę z powrotem do WGS84
            path_wgs84 = []
            for node in path:
                node_gdf = gpd.GeoDataFrame(
                    geometry=[Point(node[0], node[1])],  # x, y w EPSG:2180
                    crs="EPSG:2180"
                ).to_crs(epsg=4326)
                path_wgs84.append((node_gdf.geometry.y[0], node_gdf.geometry.x[0]))  # lat, lon
            
            return path_wgs84
            
        except nx.NetworkXNoPath:
            logger.warning(f"Nie znaleziono ścieżki między {start_point} a {end_point}")
            return [start_point, end_point]
        except Exception as e:
            logger.warning(f"Błąd podczas wyszukiwania ścieżki: {str(e)}")
            return [start_point, end_point]

    def _find_nearest_point_in_graph(self, point: Tuple[float, float], max_distance: float = 1000) -> Optional[Tuple[float, float]]:
        """
        Znajduje najbliższy punkt w grafie sieci ulic - ZOPTYMALIZOWANA WERSJA.
        
        Args:
            point: Punkt do znalezienia (lat, lon)
            max_distance: Maksymalna odległość w metrach
            
        Returns:
            Optional[Tuple[float, float]]: Najbliższy punkt lub None jeśli nie znaleziono
        """
        # Sprawdź czy spatial index istnieje
        if self.spatial_index is None:
            logger.warning("Spatial index nie istnieje - graf może być pusty")
            return None
        
        # Sprawdź cache
        cache_key = (round(point[0], 6), round(point[1], 6), max_distance)
        if cache_key in self._nearest_point_cache:
            return self._nearest_point_cache[cache_key]
        
        try:
            # Konwersja punktu wejściowego do EPSG:2180
            point_gdf = gpd.GeoDataFrame(
                geometry=[Point(point[1], point[0])],  # (lon, lat)
                crs="EPSG:4326"
            ).to_crs(epsg=2180)
            point_epsg2180 = (point_gdf.geometry.x[0], point_gdf.geometry.y[0])  # (x, y)
        except Exception as e:
            logger.warning(f"Błąd konwersji punktu do EPSG:2180: {str(e)}")
            return None
        
        # OPTYMALIZACJA: Użyj spatial index zamiast iteracji przez wszystkie węzły
        try:
            # Znajdź 10 najbliższych węzłów
            distances, indices = self.spatial_index.query(point_epsg2180, k=min(10, len(self.graph_nodes_list)))
            
            # Sprawdź czy którykolwiek jest w zasięgu
            for dist, idx in zip(distances, indices):
                if dist <= max_distance:
                    # Konwertuj wybrany węzeł z powrotem do WGS84
                    node = self.graph_nodes_list[idx]
                    try:
                        node_gdf = gpd.GeoDataFrame(
                            geometry=[Point(node[0], node[1])],  # (x, y) w EPSG:2180
                            crs="EPSG:2180"
                        ).to_crs(epsg=4326)
                        nearest_point_wgs84 = (node_gdf.geometry.y[0], node_gdf.geometry.x[0])  # (lat, lon)
                        
                        # Zapisz w cache
                        self._nearest_point_cache[cache_key] = nearest_point_wgs84
                        return nearest_point_wgs84
                    except Exception as e:
                        logger.warning(f"Błąd konwersji węzła do WGS84: {str(e)}")
                        continue
            
            logger.debug(f"Nie znaleziono punktu w zasięgu {max_distance}m")
            self._nearest_point_cache[cache_key] = None
            return None
            
        except Exception as e:
            logger.warning(f"Błąd podczas wyszukiwania w spatial index: {str(e)}")
            self._nearest_point_cache[cache_key] = None
            return None

    def _create_connected_route(self, stops: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Tworzy połączoną trasę z listy przystanków używając rzeczywistych dróg.
        
        Args:
            stops: Lista przystanków (lat, lon) w WGS84
            
        Returns:
            List[Tuple[float, float]]: Połączona trasa jako lista punktów
        """
        if len(stops) < 2:
            return stops
        
        connected_route = [stops[0]]  # Rozpocznij od pierwszego przystanku
        
        for i in range(len(stops) - 1):
            current_stop = stops[i]
            next_stop = stops[i + 1]
            
            # Znajdź ścieżkę między bieżącym a następnym przystankiem
            path = self._find_connecting_path(current_stop, next_stop)
            
            # Dodaj punkty ścieżki (pomijając pierwszy punkt, bo już jest w trasie)
            if len(path) > 1:
                connected_route.extend(path[1:])
            else:
                # Jeśli nie znaleziono ścieżki, po prostu połącz punkty bezpośrednio
                connected_route.append(next_stop)
        
        return connected_route

    def _ensure_unique_stops(self, route: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Zapewnia unikatowość przystanków w trasie i globalnie w systemie.
        
        Args:
            route: Lista punktów trasy
            
        Returns:
            List[Tuple[float, float]]: Trasa z unikatowymi przystankami
        """
        # Konwertuj przystanki do tupli z zaokrąglonymi współrzędnymi dla porównania
        def normalize_coords(lat, lon):
            return (round(lat, 6), round(lon, 6))
        
        unique_route = []
        seen_in_route = set()
        
        for point in route:
            normalized = normalize_coords(point[0], point[1])
            
            # Sprawdź czy punkt już występuje w tej trasie
            if normalized in seen_in_route:
                continue
                
            # Sprawdź czy punkt jest już używany w innej trasie w systemie
            if normalized in self.used_stops:
                # Znajdź alternatywny przystanek w pobliżu
                alternative = self._find_alternative_stop(point, min_distance=50)
                if alternative:
                    normalized_alt = normalize_coords(alternative[0], alternative[1])
                    if normalized_alt not in seen_in_route and normalized_alt not in self.used_stops:
                        unique_route.append(alternative)
                        seen_in_route.add(normalized_alt)
                # Jeśli nie znaleziono alternatywy, pomijamy ten punkt
            else:
                unique_route.append(point)
                seen_in_route.add(normalized)
                
        return unique_route

    def _find_alternative_stop(self, original_stop: Tuple[float, float], min_distance: float = 50) -> Optional[Tuple[float, float]]:
        """
        Znajduje alternatywny przystanek w pobliżu oryginalnego.
        
        Args:
            original_stop: Oryginalny przystanek (lat, lon)
            min_distance: Minimalna odległość od oryginalnego przystanku w metrach
            
        Returns:
            Optional[Tuple[float, float]]: Alternatywny przystanek lub None
        """
        # Sprawdź wszystkie dostępne przystanki
        valid_stops = [(row.geometry.y, row.geometry.x) for _, row in self.stops_df.iterrows()]
        
        for stop in valid_stops:
            distance = self._calculate_distance(original_stop, stop, is_wgs84=True)
            normalized = (round(stop[0], 6), round(stop[1], 6))
            
            # Sprawdź czy przystanek jest w odpowiedniej odległości i nie jest używany
            if (min_distance <= distance <= min_distance * 3 and 
                normalized not in self.used_stops):
                return stop
                
        return None

    def optimize_route(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        num_stops: int,
        max_iterations: int = 1000
    ) -> Tuple[List[Tuple[float, float]], float]:
        """
        Optymalizuje trasę tramwajową.
        
        Args:
            start_point: Punkt początkowy trasy
            end_point: Punkt końcowy trasy
            num_stops: Liczba przystanków
            max_iterations: Maksymalna liczba iteracji algorytmu
            
        Returns:
            Tuple[List[Tuple[float, float]], float]: Najlepsza trasa i jej ocena
        """
        # Znalezienie najbliższych punktów w sieci ulic
        logger.info("Szukam najbliższych punktów w grafie...")
        start_time = time.time()
        start_point_in_graph = self._find_nearest_point_in_graph(start_point, max_distance=1000)
        end_point_in_graph = self._find_nearest_point_in_graph(end_point, max_distance=1000)
        logger.info(f"Znaleziono punkty w grafie w {time.time() - start_time:.2f}s")
        
        if start_point_in_graph is None or end_point_in_graph is None:
            raise ValueError("Nie można znaleźć punktów w sieci ulic")
        
        logger.info(f"Znaleziono punkty w sieci ulic: {start_point_in_graph} -> {end_point_in_graph}")
        
        best_route = None
        best_score = float('-inf')
        
        logger.info(f"Rozpoczynam {max_iterations} iteracji optymalizacji...")
        
        # Pobierz listę wszystkich dostępnych przystanków
        if self.stops_df is not None:
            valid_stops = [(row.geometry.y, row.geometry.x) for _, row in self.stops_df.iterrows()]
        else:
            logger.warning("Brak danych o przystankach - używam punktów grafu")
            # Konwertuj przykładowe węzły grafu do WGS84 jako backup
            sample_nodes = list(self.street_graph.nodes())[:100]  # Tylko pierwsze 100 dla wydajności
            valid_stops = []
            for node in sample_nodes:
                try:
                    node_gdf = gpd.GeoDataFrame(
                        geometry=[Point(node[0], node[1])], crs="EPSG:2180"
                    ).to_crs(epsg=4326)
                    valid_stops.append((node_gdf.geometry.y[0], node_gdf.geometry.x[0]))
                except:
                    continue
        
        for iteration in range(max_iterations):
            iteration_start = time.time()
            
            # Logowanie postępu co 10 iteracji (zwiększona częstotliwość)
            if iteration % 10 == 0:
                logger.info(f"Iteracja {iteration}/{max_iterations}, najlepszy wynik: {best_score:.3f}")
            
            # POPRAWKA: Generuj różnorodne trasy
            route_generation_start = time.time()
            if iteration % 10 == 0 or len(valid_stops) < 10:
                # Co 10. iteracja: używaj oryginalnych punktów (deterministic baseline)
                route = self._generate_random_route(start_point_in_graph, end_point_in_graph, num_stops)
            else:
                # Pozostałe iteracje: używaj losowych punktów startowych/końcowych z przystanków
                random_start = random.choice(valid_stops)
                random_end = random.choice(valid_stops)
                
                # Znajdź punkty w grafie dla losowych przystanków
                random_start_in_graph = self._find_nearest_point_in_graph(random_start)
                random_end_in_graph = self._find_nearest_point_in_graph(random_end)
                
                if random_start_in_graph and random_end_in_graph:
                    route = self._generate_random_route(random_start_in_graph, random_end_in_graph, num_stops)
                else:
                    # Fallback do oryginalnych punktów
                    route = self._generate_random_route(start_point_in_graph, end_point_in_graph, num_stops)
            
            route_generation_time = time.time() - route_generation_start
            if iteration % 10 == 0:
                logger.info(f"Generowanie trasy zajęło: {route_generation_time:.2f}s")
            
            # Obliczanie oceny trasy
            score_calculation_start = time.time()
            density_score = self.calculate_density_score(route)
            distance_score = self.calculate_distance_score(route)
            
            total_score = (
                self.population_weight * density_score +
                self.distance_weight * distance_score
            )
            score_calculation_time = time.time() - score_calculation_start
            
            if iteration % 10 == 0:
                logger.info(f"Obliczanie oceny zajęło: {score_calculation_time:.2f}s")
            
            if total_score > best_score:
                best_score = total_score
                best_route = route
                logger.info(f"Znaleziono lepszą trasę w iteracji {iteration}: wynik {best_score:.3f}")
            
            iteration_time = time.time() - iteration_start
            if iteration % 10 == 0:
                logger.info(f"Całkowity czas iteracji {iteration}: {iteration_time:.2f}s")
                
        logger.info(f"Optymalizacja zakończona po {max_iterations} iteracjach. Najlepszy wynik: {best_score:.3f}")
        return best_route, best_score
    
    def _generate_random_route(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        num_stops: int
    ) -> List[Tuple[float, float]]:
        """
        Generuje losową trasę między punktami startowym i końcowym.
        
        Args:
            start_point: Punkt początkowy (lat, lon)
            end_point: Punkt końcowy (lat, lon)
            num_stops: Liczba przystanków
            
        Returns:
            List[Tuple[float, float]]: Wygenerowana trasa
        """
        try:
            # Znajdź najbliższe węzły w grafie - używamy cache jeśli punkty się nie zmieniły
            start_node_in_graph = None
            end_node_in_graph = None
            min_start_dist = float('inf')
            min_end_dist = float('inf')
            
            # Konwertuj punkty start i end do EPSG:2180 raz na początku
            try:
                start_gdf = gpd.GeoDataFrame(
                    geometry=[Point(start_point[1], start_point[0])],  # (lon, lat)
                    crs="EPSG:4326"
                ).to_crs(epsg=2180)
                start_epsg2180 = (start_gdf.geometry.x[0], start_gdf.geometry.y[0])
                
                end_gdf = gpd.GeoDataFrame(
                    geometry=[Point(end_point[1], end_point[0])],  # (lon, lat)
                    crs="EPSG:4326"
                ).to_crs(epsg=2180)
                end_epsg2180 = (end_gdf.geometry.x[0], end_gdf.geometry.y[0])
            except Exception as e:
                logger.error(f"Błąd konwersji punktów do EPSG:2180: {str(e)}")
                return [start_point, end_point]
            
            # Optymalizacja: przerwij wyszukiwanie gdy znajdziemy bardzo blisko węzły
            found_good_start = False
            found_good_end = False
            
            for node in self.street_graph.nodes():
                # Jeśli już znaleźliśmy dobre węzły, nie szukaj dalej
                if found_good_start and found_good_end:
                    break
                    
                if not found_good_start:
                    start_dist = self._calculate_distance(start_epsg2180, node, is_wgs84=False)
                    if start_dist < min_start_dist and start_dist > 0:
                        min_start_dist = start_dist
                        start_node_in_graph = node
                        # Jeśli znaleziono bardzo blisko (< 50m), to wystarczy
                        if start_dist < 50:
                            found_good_start = True
                
                if not found_good_end:
                    end_dist = self._calculate_distance(end_epsg2180, node, is_wgs84=False)
                    if end_dist < min_end_dist and end_dist > 0:
                        min_end_dist = end_dist
                        end_node_in_graph = node
                        # Jeśli znaleziono bardzo blisko (< 50m), to wystarczy
                        if end_dist < 50:
                            found_good_end = True
            
            if start_node_in_graph is None or end_node_in_graph is None:
                logger.error("Nie można znaleźć węzłów w grafie")
                return [start_point, end_point]
            
            # Znajdź najkrótszą ścieżkę
            try:
                path = nx.shortest_path(
                    self.street_graph,
                    start_node_in_graph,
                    end_node_in_graph,
                    weight='weight'
                )
            except nx.NetworkXNoPath:
                logger.warning("Nie znaleziono ścieżki między punktami")
                return [start_point, end_point]
            
            # Optymalizacja: jeśli ścieżka jest krótka, konwertuj tylko wybrane punkty
            if len(path) <= num_stops:
                # Konwertuj wszystkie punkty
                wgs84_path = []
                for node in path:
                    try:
                        node_gdf = gpd.GeoDataFrame(
                            geometry=[Point(node[0], node[1])],  # (x, y) w EPSG:2180
                            crs="EPSG:2180"
                        ).to_crs(epsg=4326)
                        lat = node_gdf.geometry.y[0]
                        lon = node_gdf.geometry.x[0]
                        wgs84_path.append((lat, lon))
                    except Exception as e:
                        logger.warning(f"Błąd konwersji węzła {node} do WGS84: {str(e)}")
                        return [start_point, end_point]
                return wgs84_path
            else:
                # Wybierz równomiernie rozłożone punkty PRZED konwersją
                indices = np.linspace(0, len(path) - 1, num_stops, dtype=int)
                selected_nodes = [path[i] for i in indices]
                
                # Konwertuj tylko wybrane punkty
                wgs84_path = []
                for node in selected_nodes:
                    try:
                        node_gdf = gpd.GeoDataFrame(
                            geometry=[Point(node[0], node[1])],  # (x, y) w EPSG:2180
                            crs="EPSG:2180"
                        ).to_crs(epsg=4326)
                        lat = node_gdf.geometry.y[0]
                        lon = node_gdf.geometry.x[0]
                        wgs84_path.append((lat, lon))
                    except Exception as e:
                        logger.warning(f"Błąd konwersji węzła {node} do WGS84: {str(e)}")
                        return [start_point, end_point]
                return wgs84_path
            
        except Exception as e:
            logger.error(f"Błąd podczas generowania trasy: {str(e)}")
            return [start_point, end_point]

    def _calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """Oblicza kąt między trzema punktami."""
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _calculate_total_length(self, route: List[Tuple[float, float]]) -> float:
        """Oblicza całkowitą długość trasy w metrach."""
        total_length = 0
        for i in range(len(route) - 1):
            total_length += self._calculate_distance(route[i], route[i+1], is_wgs84=True)
        return total_length
    
    def _is_valid_start_stop(self, point: Tuple[float, float]) -> bool:
        """Sprawdza czy punkt jest istniejącym przystankiem."""
        if self.stops_df is None:
            return True  # Jeśli nie ma przystanków, akceptuj każdy punkt
            
        try:
            point_geom = Point(point[1], point[0])  # zamiana lat,lon na lon,lat
            for _, row in self.stops_df.iterrows():
                # Zwiększona tolerancja z 0.0001 do 0.01 (około 1km)
                if point_geom.distance(row.geometry) < 0.01:
                    return True
        except Exception as e:
            logger.debug(f"Błąd podczas sprawdzania przystanku: {str(e)}")
            return True  # W przypadku błędu, akceptuj punkt
        
        # Jeśli punkt nie jest blisko żadnego przystanku, zaloguj to
        logger.debug(f"Punkt {point} nie jest blisko żadnego istniejącego przystanku")
        return False
    
    def _check_collision_with_existing_lines(self, route: List[Tuple[float, float]]) -> bool:
        """Sprawdza kolizje z istniejącymi liniami tramwajowymi."""
        if self.lines_df is None or len(route) < 2:
            return False
            
        try:
            route_line = LineString([(lon, lat) for lat, lon in route])
            for _, row in self.lines_df.iterrows():
                if isinstance(row.geometry, LineString) and route_line.intersects(row.geometry):
                    return True
        except Exception as e:
            logger.debug(f"Błąd podczas sprawdzania kolizji z liniami: {str(e)}")
        return False
    
    def _check_collision_with_buildings(self, route: List[Tuple[float, float]]) -> bool:
        """Sprawdza kolizje z budynkami."""
        if self.buildings_df is None or len(route) < 2:
            return False
            
        try:
            route_line = LineString([(lon, lat) for lat, lon in route])
            for _, row in self.buildings_df.iterrows():
                if route_line.intersects(row.geometry):
                    return True
        except Exception as e:
            logger.debug(f"Błąd podczas sprawdzania kolizji z budynkami: {str(e)}")
        return False
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float], is_wgs84: bool = True) -> float:
        """
        Bezpiecznie oblicza odległość między dwoma punktami w metrach.
        
        Args:
            point1: Pierwszy punkt - (lat, lon) jeśli is_wgs84=True, (x, y) w EPSG:2180 jeśli is_wgs84=False
            point2: Drugi punkt - (lat, lon) jeśli is_wgs84=True, (x, y) w EPSG:2180 jeśli is_wgs84=False
            is_wgs84: Czy współrzędne są w układzie WGS84 (domyślnie True)
            
        Returns:
            float: Odległość w metrach lub 0 w przypadku błędu
        """
        try:
            # Sprawdzenie czy współrzędne nie są None lub NaN
            if any(p is None or np.isnan(p) for p in point1 + point2):
                logger.warning(f"Współrzędne zawierają None lub NaN: {point1}, {point2}")
                return 0
            
            if is_wgs84:
                # Sprawdzenie czy współrzędne WGS84 są prawidłowe
                if not (self._validate_coordinates(point1, is_wgs84=True) and 
                       self._validate_coordinates(point2, is_wgs84=True)):
                    logger.warning(f"Nieprawidłowe współrzędne WGS84: {point1} lub {point2}")
                    return 0
                
                # Konwersja z WGS84 do EPSG:2180
                p1_gdf = gpd.GeoDataFrame(
                    geometry=[Point(point1[1], point1[0])],  # (lon, lat)
                    crs="EPSG:4326"
                ).to_crs(epsg=2180)
                
                p2_gdf = gpd.GeoDataFrame(
                    geometry=[Point(point2[1], point2[0])],  # (lon, lat)
                    crs="EPSG:4326"
                ).to_crs(epsg=2180)
                
                p1 = p1_gdf.geometry[0]
                p2 = p2_gdf.geometry[0]
                
            else:
                # Sprawdzenie czy współrzędne EPSG:2180 są prawidłowe
                if not (self._validate_coordinates(point1, is_wgs84=False) and 
                       self._validate_coordinates(point2, is_wgs84=False)):
                    logger.warning(f"Nieprawidłowe współrzędne EPSG:2180: {point1} lub {point2}")
                    return 0
                
                # Tworzenie punktów bezpośrednio w układzie EPSG:2180
                p1 = Point(point1[0], point1[1])  # (x, y)
                p2 = Point(point2[0], point2[1])  # (x, y)
            
            # Obliczanie odległości w metrach
            distance = float(p1.distance(p2))
            
            # Sprawdzenie czy odległość jest prawidłowa
            if np.isnan(distance) or np.isinf(distance):
                logger.warning(f"Nieprawidłowa odległość: {distance}")
                return 0
                
            # Dodanie minimalnej odległości, aby uniknąć ostrzeżeń
            if distance < 0.001:  # 1mm
                return 0.001
                
            return distance
            
        except Exception as e:
            logger.warning(f"Błąd podczas obliczania odległości: {str(e)}")
            return 0

    def _is_valid_route(self, route: List[Tuple[float, float]], is_simplified: bool = False) -> bool:
        """
        Sprawdza czy trasa spełnia wszystkie ograniczenia.
        
        Args:
            route: Trasa do sprawdzenia
            is_simplified: Czy trasa jest częścią uproszczonej populacji
            
        Returns:
            bool: True jeśli trasa spełnia ograniczenia
        """
        try:
            # Sprawdzenie długości trasy
            if not is_simplified:
                if not (self.constraints.min_route_length <= len(route) <= self.constraints.max_route_length):
                    logger.debug(f"Nieprawidłowa długość trasy: {len(route)}")
                    return False
            else:
                if len(route) < 2:  # Dla uproszczonej populacji minimum 2 przystanki
                    return False
                
            # Sprawdzenie całkowitej długości
            total_length = 0
            for i in range(len(route) - 1):
                dist = self._calculate_distance(route[i], route[i + 1], is_wgs84=True)
                if dist == 0:  # Błąd podczas obliczania odległości
                    return False
                total_length += dist
                
            if not is_simplified:
                if not (self.constraints.min_total_length <= total_length <= self.constraints.max_total_length):
                    logger.debug(f"Nieprawidłowa całkowita długość trasy: {total_length}m")
                    return False
                
            # Sprawdzenie początkowego przystanku - czasowo wyłączone dla debugowania
            # if not self._is_valid_start_stop(route[0]):
            #     logger.debug("Nieprawidłowy przystanek początkowy")
            #     return False
                
            # Sprawdzenie odległości między przystankami i kątów
            for i in range(len(route) - 1):
                dist = self._calculate_distance(route[i], route[i + 1], is_wgs84=True)
                if not is_simplified:
                    if not (self.constraints.min_distance_between_stops <= dist <= self.constraints.max_distance_between_stops):
                        logger.debug(f"Nieprawidłowa odległość między przystankami: {dist}m")
                        return False
                else:
                    if dist == 0:  # Dla uproszczonej populacji sprawdzamy tylko czy odległość jest prawidłowa
                        return False
                    
                if i > 0 and not is_simplified:
                    angle = self._calculate_angle(route[i-1], route[i], route[i+1])
                    if angle > self.constraints.max_angle:
                        logger.debug(f"Nieprawidłowy kąt zakrętu: {angle}°")
                        return False
            
            # Sprawdzenie kolizji z istniejącymi liniami
            if self._check_collision_with_existing_lines(route):
                logger.debug("Wykryto kolizję z istniejącymi liniami")
                return False
                
            # Sprawdzenie kolizji z budynkami - czasowo wyłączone dla debugowania
            # if self._check_collision_with_buildings(route):
            #     logger.debug("Wykryto kolizję z budynkami")
            #     return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Błąd podczas sprawdzania trasy: {str(e)}")
            return False
    
    def _evaluate_route(self, route: List[Tuple[float, float]]) -> float:
        """Ocenia jakość trasy."""
        if not self._is_valid_route(route):
            return float('-inf')
            
        # Obliczanie gęstości zabudowy
        density_score = self.calculate_density_score(route)
        
        # Obliczanie odległości między przystankami
        distance_score = self.calculate_distance_score(route)
        
        # Obliczanie prostoty trasy (minimalizacja kątów zakrętu)
        angle_score = self.calculate_angle_score(route)
        
        # Łączna ocena - wszystkie składniki są znormalizowane do 0-1
        score = (self.population_weight * density_score +
                self.distance_weight * distance_score +
                self.angle_weight * angle_score)
                
        return score
    
    def _create_initial_population(self) -> List[List[Tuple[float, float]]]:
        """Tworzy początkową populację tras."""
        population = []
        # Używamy oryginalnego stops_df w WGS84, nie projected
        valid_stops = [(row.geometry.y, row.geometry.x) for _, row in self.stops_df.iterrows()]
        
        logger.info(f"Liczba dostępnych przystanków: {len(valid_stops)}")
        logger.info(f"Ograniczenia: min_route_length={self.constraints.min_route_length}, "
                   f"max_route_length={self.constraints.max_route_length}")
        
        attempts = 0
        max_attempts = self.population_size * 50  # Zwiększamy liczbę prób
        
        while len(population) < self.population_size and attempts < max_attempts:
            try:
                # Losowa długość trasy - bardziej elastyczna
                route_length = random.randint(
                    max(2, self.constraints.min_route_length),  # Minimum 2 przystanki
                    min(self.constraints.max_route_length, len(valid_stops))
                )
                
                # Wybierz losowy punkt startowy z istniejących przystanków
                available_starts = [stop for stop in valid_stops 
                                  if (round(stop[0], 6), round(stop[1], 6)) not in self.used_stops]
                
                if not available_starts:
                    logger.warning("Brak dostępnych przystanków startowych!")
                    break
                    
                start_stop = random.choice(available_starts)
                
                # Tworzenie listy przystanków dla trasy
                route_stops = [start_stop]
                
                # Dodaj pozostałe przystanki zapewniając unikatowość
                remaining_stops = [stop for stop in valid_stops 
                                 if stop != start_stop and 
                                 (round(stop[0], 6), round(stop[1], 6)) not in self.used_stops]
                
                # Jeśli nie ma wystarczająco dużo przystanków, zmniejsz długość trasy
                if len(remaining_stops) < route_length - 1:
                    route_length = min(len(remaining_stops) + 1, len(available_starts))
                    if route_length < 2:
                        continue
                
                # Dodaj pozostałe przystanki
                if len(remaining_stops) > 0:
                    selected_stops = random.sample(remaining_stops, route_length - 1)
                    route_stops.extend(selected_stops)
                
                # Zapewnij unikatowość przystanków
                unique_stops = self._ensure_unique_stops(route_stops)
                
                if len(unique_stops) < 2:
                    continue  # Potrzebujemy przynajmniej 2 przystanków
                
                # Utwórz połączoną trasę używając rzeczywistych dróg
                connected_route = self._create_connected_route(unique_stops)
                
                # Sprawdź czy trasa jest poprawna
                if self._is_valid_route(connected_route, is_simplified=False):
                    population.append(connected_route)
                    
                    # Oznacz przystanki jako używane
                    for stop in unique_stops:
                        normalized = (round(stop[0], 6), round(stop[1], 6))
                        self.used_stops.add(normalized)
                    
                    logger.info(f"Utworzono trasę {len(population)}/{self.population_size} "
                              f"z {len(unique_stops)} przystankami")
                else:
                    logger.debug(f"Trasa nie spełnia ograniczeń")
                
            except Exception as e:
                logger.warning(f"Błąd podczas tworzenia trasy: {str(e)}")
            
            attempts += 1
            
        if len(population) == 0:
            logger.warning("Nie udało się utworzyć populacji z pełnymi ograniczeniami!")
            logger.info("Tworzę uproszczoną populację...")
            
            # Resetuj używane przystanki dla uproszczonej populacji
            self.used_stops.clear()
            
            # Tworzymy uproszczoną populację z minimalną liczbą tras
            simplified_population = []
            for _ in range(self.population_size):
                try:
                    # Wybierz dwa losowe przystanki
                    available_stops = [stop for stop in valid_stops 
                                     if (round(stop[0], 6), round(stop[1], 6)) not in self.used_stops]
                    
                    if len(available_stops) >= 2:
                        start = random.choice(available_stops)
                        available_stops.remove(start)
                        end = random.choice(available_stops)
                        
                        # Utwórz połączoną trasę
                        connected_route = self._create_connected_route([start, end])
                        
                        # Sprawdź czy trasa jest poprawna
                        if self._is_valid_route(connected_route, is_simplified=True):
                            simplified_population.append(connected_route)
                            # Oznacz przystanki jako używane
                            self.used_stops.add((round(start[0], 6), round(start[1], 6)))
                            self.used_stops.add((round(end[0], 6), round(end[1], 6)))
                        else:
                            # Jeśli trasa nie jest poprawna, dodaj ją mimo to
                            simplified_population.append(connected_route)
                    elif len(available_stops) >= 1:
                        # Jeśli nie ma wystarczająco dużo przystanków, użyj prostej trasy
                        stop = available_stops[0]
                        simplified_population.append([stop, stop])
                except Exception as e:
                    logger.warning(f"Błąd podczas tworzenia uproszczonej trasy: {str(e)}")
                    continue
            
            logger.info(f"Utworzono uproszczoną populację o rozmiarze {len(simplified_population)}")
            return simplified_population
            
        return population
    
    def _crossover(self, parent1: List[Tuple[float, float]], 
                  parent2: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], 
                                                             List[Tuple[float, float]]]:
        """Wykonuje krzyżowanie dwóch tras zapewniając unikatowość i połączenia."""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Wyodrębnij przystanki z tras
        stops1 = self._extract_stops_from_route(parent1)
        stops2 = self._extract_stops_from_route(parent2)
        
        if len(stops1) < 2 or len(stops2) < 2:
            return parent1, parent2
            
        try:
            # Punkt krzyżowania
            point1 = random.randint(1, len(stops1) - 1)
            point2 = random.randint(1, len(stops2) - 1)
            
            # Tworzenie potomstwa
            child1_stops = stops1[:point1] + stops2[point2:]
            child2_stops = stops2[:point2] + stops1[point1:]
            
            # Zapewnij unikatowość przystanków
            child1_unique = self._ensure_unique_stops(child1_stops)
            child2_unique = self._ensure_unique_stops(child2_stops)
            
            # Sprawdź czy potomstwo ma wystarczającą liczbę przystanków
            if len(child1_unique) < 2:
                child1_unique = stops1  # Użyj oryginalnej trasy
            if len(child2_unique) < 2:
                child2_unique = stops2  # Użyj oryginalnej trasy
            
            # Utwórz połączone trasy
            child1_route = self._create_connected_route(child1_unique)
            child2_route = self._create_connected_route(child2_unique)
            
            return child1_route, child2_route
            
        except Exception as e:
            logger.warning(f"Błąd podczas krzyżowania: {str(e)}")
            return parent1, parent2
    
    def _mutate(self, route: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Wykonuje mutację trasy zapewniając unikatowość i połączenia."""
        if random.random() > self.mutation_rate:
            return route
            
        # Wyodrębnij przystanki z trasy (co pewną liczbę punktów, nie wszystkie punkty ścieżki)
        route_stops = self._extract_stops_from_route(route)
        
        if len(route_stops) < 2:
            return route
            
        mutation_type = random.choice(['swap', 'replace', 'add', 'remove'])
        
        # Używamy oryginalnego stops_df w WGS84, nie projected
        valid_stops = [(row.geometry.y, row.geometry.x) for _, row in self.stops_df.iterrows()]
        
        mutated_stops = route_stops.copy()
        
        try:
            if mutation_type == 'swap' and len(mutated_stops) >= 2:
                # Zamień dwa przystanki miejscami
                i, j = random.sample(range(len(mutated_stops)), 2)
                mutated_stops[i], mutated_stops[j] = mutated_stops[j], mutated_stops[i]
                
            elif mutation_type == 'replace':
                # Zamień jeden przystanek na nowy
                if mutated_stops:
                    # Usuń stary przystanek z used_stops
                    old_stop = mutated_stops[random.randrange(len(mutated_stops))]
                    old_normalized = (round(old_stop[0], 6), round(old_stop[1], 6))
                    self.used_stops.discard(old_normalized)
                    
                    # Znajdź nowy unikatowy przystanek
                    available_stops = [stop for stop in valid_stops 
                                     if (round(stop[0], 6), round(stop[1], 6)) not in self.used_stops]
                    
                    if available_stops:
                        new_stop = random.choice(available_stops)
                        mutated_stops[mutated_stops.index(old_stop)] = new_stop
                        # Dodaj nowy przystanek do used_stops
                        new_normalized = (round(new_stop[0], 6), round(new_stop[1], 6))
                        self.used_stops.add(new_normalized)
                    else:
                        # Przywróć stary przystanek jeśli nie ma alternatywy
                        self.used_stops.add(old_normalized)
                        
            elif mutation_type == 'add' and len(mutated_stops) < self.constraints.max_route_length:
                # Dodaj nowy przystanek
                available_stops = [stop for stop in valid_stops 
                                 if (round(stop[0], 6), round(stop[1], 6)) not in self.used_stops]
                
                if available_stops:
                    new_stop = random.choice(available_stops)
                    insert_position = random.randint(0, len(mutated_stops))
                    mutated_stops.insert(insert_position, new_stop)
                    # Dodaj do used_stops
                    normalized = (round(new_stop[0], 6), round(new_stop[1], 6))
                    self.used_stops.add(normalized)
                    
            elif mutation_type == 'remove' and len(mutated_stops) > self.constraints.min_route_length:
                # Usuń przystanek
                if mutated_stops:
                    removed_stop = mutated_stops.pop(random.randrange(len(mutated_stops)))
                    # Usuń z used_stops
                    normalized = (round(removed_stop[0], 6), round(removed_stop[1], 6))
                    self.used_stops.discard(normalized)
        
        except Exception as e:
            logger.warning(f"Błąd podczas mutacji: {str(e)}")
            return route
        
        # Zapewnij unikatowość i utwórz połączoną trasę
        unique_stops = self._ensure_unique_stops(mutated_stops)
        
        if len(unique_stops) < 2:
            return route  # Zwróć oryginalną trasę jeśli mutacja się nie powiodła
            
        try:
            connected_route = self._create_connected_route(unique_stops)
            return connected_route
        except Exception as e:
            logger.warning(f"Błąd podczas tworzenia połączonej trasy po mutacji: {str(e)}")
            return route

    def _extract_stops_from_route(self, route: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Wyodrębnia główne przystanki z trasy (pomijając punkty pośrednie ścieżki).
        
        Args:
            route: Pełna trasa z punktami ścieżki
            
        Returns:
            List[Tuple[float, float]]: Lista głównych przystanków
        """
        if len(route) <= 2:
            return route
            
        # Prosty algorytm: weź pierwszy, ostatni i co kilka punktów pośrednich
        stops = [route[0]]  # Pierwszy punkt
        
        # Dodaj punkty pośrednie co określoną liczbę kroków
        step = max(1, len(route) // 10)  # Około 10 przystanków max
        for i in range(step, len(route) - 1, step):
            stops.append(route[i])
            
        # Dodaj ostatni punkt jeśli nie jest identyczny z pierwszym
        if route[-1] != route[0]:
            stops.append(route[-1])
            
        return stops

    def optimize(self) -> Tuple[List[Tuple[float, float]], float]:
        """
        Wykonuje optymalizację trasy.
        
        Returns:
            Tuple[List[Tuple[float, float]], float]: Najlepsza znaleziona trasa i jej ocena
        """
        population = self._create_initial_population()
        best_route = None
        best_score = float('-inf')
        
        for generation in range(self.generations):
            # Ocena populacji
            scores = [self._evaluate_route(route) for route in population]
            
            # Aktualizacja najlepszej trasy
            max_score_idx = np.argmax(scores)
            if scores[max_score_idx] > best_score:
                best_score = scores[max_score_idx]
                best_route = population[max_score_idx]
            
            # Selekcja
            selected_indices = np.argsort(scores)[-self.population_size//2:]
            selected = [population[i] for i in selected_indices]
            
            # Tworzenie nowej populacji
            new_population = selected.copy()
            
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self._crossover(parent1, parent2)
                
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                if self._is_valid_route(child1):
                    new_population.append(child1)
                if self._is_valid_route(child2) and len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population
            
            logger.info(f"Pokolenie {generation + 1}/{self.generations}, "
                       f"najlepszy wynik: {best_score:.2f}")
        
        return best_route, best_score

    def reset_used_stops(self):
        """Resetuje set używanych przystanków."""
        self.used_stops.clear()
        logger.info("Zresetowano używane przystanki")

    def optimize_multiple_routes(self, num_routes: int = 3) -> List[Tuple[List[Tuple[float, float]], float]]:
        """
        Optymalizuje wiele tras jednocześnie zapewniając unikatowość przystanków.
        
        Args:
            num_routes: Liczba tras do optymalizacji
            
        Returns:
            List[Tuple[List[Tuple[float, float]], float]]: Lista tras i ich ocen
        """
        routes = []
        
        for route_idx in range(num_routes):
            logger.info(f"Optymalizacja trasy {route_idx + 1}/{num_routes}")
            
            # Optymalizuj jedną trasę
            best_route, best_score = self.optimize()
            
            if best_route is not None:
                routes.append((best_route, best_score))
                logger.info(f"Zakończono trasę {route_idx + 1} z wynikiem: {best_score:.2f}")
                
                # Oznacz przystanki tej trasy jako używane dla następnych tras
                route_stops = self._extract_stops_from_route(best_route)
                for stop in route_stops:
                    normalized = (round(stop[0], 6), round(stop[1], 6))
                    self.used_stops.add(normalized)
            else:
                logger.warning(f"Nie udało się znaleźć trasy {route_idx + 1}")
        
        return routes

    def _create_spatial_index(self):
        """Tworzy spatial index dla szybkiego wyszukiwania najbliższych węzłów."""
        # Konwertuj węzły grafu do listy współrzędnych
        self.graph_nodes_list = list(self.street_graph.nodes())
        
        # Sprawdź czy graf nie jest pusty
        if len(self.graph_nodes_list) == 0:
            logger.error("Graf jest pusty! Nie można utworzyć spatial index.")
            self.spatial_index = None
            return
        
        self.graph_nodes_coords = np.array([(node[0], node[1]) for node in self.graph_nodes_list])
        
        # Utwórz KDTree dla szybkiego wyszukiwania
        self.spatial_index = cKDTree(self.graph_nodes_coords)
        logger.info(f"Spatial index utworzony dla {len(self.graph_nodes_list)} węzłów")

    def _find_top_density_stops(self, top_n: int = 5) -> List[Tuple[float, float]]:
        """
        Znajduje przystanki o najwyższej gęstości zabudowy w promieniu 300m.
        
        Args:
            top_n: Liczba przystanków do zwrócenia
            
        Returns:
            List[Tuple[float, float]]: Lista współrzędnych (lat, lon) najlepszych przystanków
        """
        logger.info(f"Obliczanie gęstości zabudowy dla {len(self.stops_df)} przystanków...")
        
        stop_densities = []
        radius = 300  # 300m promień
        
        for idx, stop in self.stops_df.iterrows():
            # Konwertuj przystanek do EPSG:2180
            stop_projected = gpd.GeoDataFrame(
                geometry=[stop.geometry],
                crs="EPSG:4326"
            ).to_crs(epsg=2180).geometry[0]
            
            # Znajdź budynki w promieniu 300m
            buildings_nearby = self.buildings_projected[
                self.buildings_projected.geometry.distance(stop_projected) <= radius
            ]
            
            # Oblicz gęstość jako liczba budynków / powierzchnia koła
            density = len(buildings_nearby) / (np.pi * radius**2) * 1000000  # na km²
            
            stop_densities.append({
                'coords': (stop.geometry.y, stop.geometry.x),  # lat, lon
                'density': density,
                'buildings_count': len(buildings_nearby)
            })
        
        # Sortuj według gęstości
        stop_densities.sort(key=lambda x: x['density'], reverse=True)
        
        # Loguj TOP przystanki
        logger.info("TOP przystanki według gęstości zabudowy:")
        for i, stop in enumerate(stop_densities[:top_n]):
            logger.info(f"  {i+1}. Gęstość: {stop['density']:.1f} budynków/km², "
                       f"Budynki: {stop['buildings_count']}, Coords: {stop['coords']}")
        
        return [stop['coords'] for stop in stop_densities[:top_n]] 

    def optimize_multiple_routes_fast(self, num_routes: int = 3) -> List[Tuple[List[Tuple[float, float]], float]]:
        """
        SZYBKA optymalizacja wielu tras - zredukowane parametry dla praktycznego użycia.
        
        Args:
            num_routes: Liczba tras do optymalizacji
            
        Returns:
            List[Tuple[List[Tuple[float, float]], float]]: Lista tras i ich ocen
        """
        routes = []
        
        # Zapisz oryginalne parametry
        original_population_size = self.population_size
        original_generations = self.generations
        
        # DRASTYCZNA REDUKCJA PARAMETRÓW DLA SZYBKOŚCI
        self.population_size = 20  # Zamiast 100
        self.generations = 15      # Zamiast 50
        
        logger.info(f"🚀 SZYBKA optymalizacja {num_routes} tras:")
        logger.info(f"   Populacja: {self.population_size} (było: {original_population_size})")
        logger.info(f"   Pokolenia: {self.generations} (było: {original_generations})")
        logger.info(f"   Łączne ewaluacje: {self.population_size * self.generations * num_routes}")
        
        start_total = time.time()
        
        for route_idx in range(num_routes):
            logger.info(f"Optymalizacja trasy {route_idx + 1}/{num_routes}")
            route_start = time.time()
            
            # Optymalizuj jedną trasę z early stopping
            best_route, best_score = self._optimize_with_early_stopping()
            
            route_time = time.time() - route_start
            
            if best_route is not None:
                routes.append((best_route, best_score))
                logger.info(f"✅ Trasa {route_idx + 1} gotowa w {route_time:.1f}s, wynik: {best_score:.2f}")
                
                # Oznacz przystanki tej trasy jako używane
                route_stops = self._extract_stops_from_route(best_route)
                for stop in route_stops:
                    normalized = (round(stop[0], 6), round(stop[1], 6))
                    self.used_stops.add(normalized)
                    
                logger.info(f"   Dodano {len(route_stops)} przystanków do listy używanych")
            else:
                logger.warning(f"❌ Nie udało się znaleźć trasy {route_idx + 1}")
        
        total_time = time.time() - start_total
        
        # Przywróć oryginalne parametry
        self.population_size = original_population_size
        self.generations = original_generations
        
        logger.info(f"🏁 Zakończono w {total_time:.1f}s (średnio {total_time/num_routes:.1f}s/trasa)")
        
        return routes
    
    def _optimize_with_early_stopping(self, patience: int = 5) -> Tuple[List[Tuple[float, float]], float]:
        """
        Optymalizacja z early stopping - zatrzymuje się gdy brak poprawy.
        
        Args:
            patience: Liczba pokoleń bez poprawy po której zatrzymać
            
        Returns:
            Tuple[List[Tuple[float, float]], float]: Najlepsza trasa i ocena
        """
        population = self._create_initial_population()
        best_route = None
        best_score = float('-inf')
        generations_without_improvement = 0
        
        for generation in range(self.generations):
            # Ocena populacji
            scores = [self._evaluate_route(route) for route in population]
            
            # Sprawdź czy jest poprawa
            max_score_idx = np.argmax(scores)
            current_best_score = scores[max_score_idx]
            
            if current_best_score > best_score:
                best_score = current_best_score
                best_route = population[max_score_idx]
                generations_without_improvement = 0
                logger.debug(f"🎯 Poprawa w pokoleniu {generation + 1}: {best_score:.3f}")
            else:
                generations_without_improvement += 1
            
            # Early stopping
            if generations_without_improvement >= patience:
                logger.info(f"⏹️ Early stopping po {generation + 1} pokoleniach (brak poprawy przez {patience})")
                break
            
            # Selekcja (tylko najlepsze 50%)
            selected_indices = np.argsort(scores)[-self.population_size//2:]
            selected = [population[i] for i in selected_indices]
            
            # Tworzenie nowej populacji
            new_population = selected.copy()
            
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self._crossover(parent1, parent2)
                
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                if self._is_valid_route(child1):
                    new_population.append(child1)
                if self._is_valid_route(child2) and len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population
        
        return best_route, best_score