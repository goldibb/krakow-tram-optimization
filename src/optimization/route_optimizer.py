import numpy as np
from typing import List, Tuple, Dict, Optional
import random
from dataclasses import dataclass
import logging
from .density_calculator import DensityCalculator
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
import networkx as nx
from shapely.ops import unary_union
from scipy.spatial import distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RouteConstraints:
    min_distance_between_stops: float = 300  # minimalna odległość między przystankami w metrach
    max_distance_between_stops: float = 1000  # maksymalna odległość między przystankami w metrach
    max_angle: float = 45  # maksymalny kąt zakrętu w stopniach
    min_route_length: int = 5  # minimalna liczba przystanków
    max_route_length: int = 20  # maksymalna liczba przystanków
    min_total_length: float = 2000  # minimalna długość całkowita trasy w metrach
    max_total_length: float = 10000  # maksymalna długość całkowita trasy w metrach
    min_distance_from_buildings: float = 5  # minimalna odległość od budynków w metrach

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
        min_stop_distance: float = 300,  # minimalna odległość między przystankami w metrach
        max_stop_distance: float = 800,  # maksymalna odległość między przystankami w metrach
        population_weight: float = 0.7,  # waga dla kryterium gęstości zaludnienia
        distance_weight: float = 0.3,    # waga dla kryterium odległości
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
            min_stop_distance: Minimalna odległość między przystankami w metrach
            max_stop_distance: Maksymalna odległość między przystankami w metrach
            population_weight: Waga dla kryterium gęstości zaludnienia
            distance_weight: Waga dla kryterium odległości
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
        self.min_stop_distance = min_stop_distance
        self.max_stop_distance = max_stop_distance
        self.population_weight = population_weight
        self.distance_weight = distance_weight
        
        # Transformacja do układu współrzędnych rzutowanych (EPSG:2180 dla Polski)
        self.buildings_projected = buildings_df.to_crs(epsg=2180)
        self.streets_projected = streets_df.to_crs(epsg=2180)
        if stops_df is not None:
            self.stops_projected = stops_df.to_crs(epsg=2180)
        if lines_df is not None:
            self.lines_projected = lines_df.to_crs(epsg=2180)
        
        # Tworzenie grafu sieci ulic
        self.street_graph = self._create_street_graph()
        
        # Przygotowanie istniejących linii i buforów
        self.existing_lines = self._prepare_existing_lines() if lines_df is not None else []
        self.buildings_buffer = self._create_buildings_buffer()
        
    def _create_street_graph(self) -> nx.Graph:
        """
        Tworzy graf sieci ulic na podstawie danych OSM.
        
        Returns:
            nx.Graph: Graf sieci ulic
        """
        G = nx.Graph()
        
        # Dodawanie węzłów (skrzyżowania)
        for idx, row in self.streets_projected.iterrows():
            coords = list(row.geometry.coords)
            for i in range(len(coords) - 1):
                G.add_edge(
                    coords[i],
                    coords[i + 1],
                    weight=distance.euclidean(coords[i], coords[i + 1])
                )
        
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
    
    def _validate_coordinates(self, point: Tuple[float, float]) -> bool:
        """
        Sprawdza czy współrzędne są prawidłowe.
        
        Args:
            point: Punkt (lat, lon) do sprawdzenia
            
        Returns:
            bool: True jeśli współrzędne są prawidłowe
        """
        try:
            lat, lon = point
            return (
                isinstance(lat, (int, float)) and 
                isinstance(lon, (int, float)) and
                -90 <= lat <= 90 and
                -180 <= lon <= 180 and
                not np.isnan(lat) and
                not np.isnan(lon)
            )
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
            # Sprawdź czy współrzędne są prawidłowe
            if not (self._validate_coordinates(route[i]) and self._validate_coordinates(route[i + 1])):
                logger.warning(f"Nieprawidłowe współrzędne w trasie: {route[i]} -> {route[i + 1]}")
                return 0
                
            try:
                # Konwersja punktów do układu EPSG:2180 dla obliczeń w metrach
                p1 = gpd.GeoDataFrame(
                    geometry=[Point(route[i][1], route[i][0])],  # (lon, lat)
                    crs="EPSG:4326"
                ).to_crs(epsg=2180)
                
                p2 = gpd.GeoDataFrame(
                    geometry=[Point(route[i + 1][1], route[i + 1][0])],  # (lon, lat)
                    crs="EPSG:4326"
                ).to_crs(epsg=2180)
                
                # Obliczanie odległości w metrach
                dist = p1.geometry[0].distance(p2.geometry[0])
                
                if dist < self.min_stop_distance:
                    return 0  # Kara za zbyt małą odległość
                if dist > self.max_stop_distance:
                    return 0  # Kara za zbyt dużą odległość
                    
                total_distance += dist
                
            except Exception as e:
                logger.warning(f"Błąd podczas obliczania odległości: {str(e)}")
                return 0
            
        # Normalizacja wyniku (im mniejsza odległość, tym lepszy wynik)
        max_possible_distance = self.max_stop_distance * (len(route) - 1)
        return 1 - (total_distance / max_possible_distance)
    
    def _find_nearest_point_in_graph(self, point: Tuple[float, float], max_distance: float = 100) -> Optional[Tuple[float, float]]:
        """
        Znajduje najbliższy punkt w grafie sieci ulic.
        
        Args:
            point: Punkt do znalezienia (lat, lon)
            max_distance: Maksymalna odległość w metrach
            
        Returns:
            Optional[Tuple[float, float]]: Najbliższy punkt lub None jeśli nie znaleziono
        """
        min_dist = float('inf')
        nearest_point = None
        
        # Konwersja punktu do układu EPSG:2180
        point_gdf = gpd.GeoDataFrame(
            geometry=[Point(point[1], point[0])],  # (lon, lat)
            crs="EPSG:4326"
        ).to_crs(epsg=2180)
        
        point_projected = (point_gdf.geometry.x[0], point_gdf.geometry.y[0])
        
        # Sprawdzenie wszystkich węzłów w grafie
        for node in self.street_graph.nodes():
            dist = distance.euclidean(point_projected, node)
            if dist < min_dist:
                min_dist = dist
                nearest_point = node
        
        if min_dist <= max_distance:
            # Konwersja z powrotem do stopni
            nearest_point_gdf = gpd.GeoDataFrame(
                geometry=[Point(nearest_point[0], nearest_point[1])],  # (lon, lat)
                crs="EPSG:2180"
            ).to_crs(epsg=4326)
            
            # Konwersja do formatu (lat, lon) i zaokrąglenie do 6 miejsc po przecinku
            lat = round(nearest_point_gdf.geometry.y[0], 6)
            lon = round(nearest_point_gdf.geometry.x[0], 6)
            
            # Znajdź dokładnie ten sam punkt w grafie
            for node in self.street_graph.nodes():
                node_lat = round(node[1], 6)
                node_lon = round(node[0], 6)
                if node_lat == lat and node_lon == lon:
                    return (node_lat, node_lon)
            
            # Jeśli nie znaleziono dokładnego dopasowania, zwróć zaokrąglony punkt
            return (lat, lon)
        
        logger.warning(f"Nie znaleziono punktu w zasięgu {max_distance}m")
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
        start_point_in_graph = self._find_nearest_point_in_graph(start_point)
        end_point_in_graph = self._find_nearest_point_in_graph(end_point)
        
        if start_point_in_graph is None or end_point_in_graph is None:
            raise ValueError("Nie można znaleźć punktów w sieci ulic")
        
        logger.info(f"Znaleziono punkty w sieci ulic: {start_point_in_graph} -> {end_point_in_graph}")
        
        best_route = None
        best_score = float('-inf')
        
        for _ in range(max_iterations):
            # Generowanie losowej trasy
            route = self._generate_random_route(start_point_in_graph, end_point_in_graph, num_stops)
            
            # Obliczanie oceny trasy
            density_score = self.calculate_density_score(route)
            distance_score = self.calculate_distance_score(route)
            
            total_score = (
                self.population_weight * density_score +
                self.distance_weight * distance_score
            )
            
            if total_score > best_score:
                best_score = total_score
                best_route = route
                
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
            # Konwersja punktów do formatu (lon, lat) dla grafu
            start_node = (start_point[1], start_point[0])  # (lon, lat)
            end_node = (end_point[1], end_point[0])  # (lon, lat)
            
            # Znajdź najbliższe węzły w grafie
            start_node_in_graph = None
            end_node_in_graph = None
            min_start_dist = float('inf')
            min_end_dist = float('inf')
            
            for node in self.street_graph.nodes():
                start_dist = distance.euclidean(start_node, node)
                end_dist = distance.euclidean(end_node, node)
                
                if start_dist < min_start_dist:
                    min_start_dist = start_dist
                    start_node_in_graph = node
                if end_dist < min_end_dist:
                    min_end_dist = end_dist
                    end_node_in_graph = node
            
            if start_node_in_graph is None or end_node_in_graph is None:
                logger.error("Nie można znaleźć węzłów w grafie")
                return [start_point, end_point]
            
            # Znajdź najkrótszą ścieżkę
            path = nx.shortest_path(
                self.street_graph,
                start_node_in_graph,
                end_node_in_graph,
                weight='weight'
            )
            
            # Konwersja z powrotem do formatu (lat, lon)
            path = [(lat, lon) for lon, lat in path]
            
            # Wybór równomiernie rozłożonych punktów na ścieżce
            if len(path) <= num_stops:
                return path
                
            indices = np.linspace(0, len(path) - 1, num_stops, dtype=int)
            return [path[i] for i in indices]
            
        except nx.NetworkXNoPath:
            logger.warning("Nie znaleziono ścieżki między punktami")
            return [start_point, end_point]
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
            total_length += distance.euclidean(route[i], route[i+1])
        return total_length
    
    def _is_valid_start_stop(self, point: Tuple[float, float]) -> bool:
        """Sprawdza czy punkt jest istniejącym przystankiem."""
        point_geom = Point(point[1], point[0])  # zamiana lat,lon na lon,lat
        for _, row in self.stops_df.iterrows():
            if point_geom.distance(row.geometry) < 0.0001:  # mała tolerancja
                return True
        return False
    
    def _check_collision_with_existing_lines(self, route: List[Tuple[float, float]]) -> bool:
        """Sprawdza kolizje z istniejącymi liniami tramwajowymi."""
        route_line = LineString([(lon, lat) for lat, lon in route])
        for _, row in self.lines_df.iterrows():
            if isinstance(row.geometry, LineString) and route_line.intersects(row.geometry):
                return True
        return False
    
    def _check_collision_with_buildings(self, route: List[Tuple[float, float]]) -> bool:
        """Sprawdza kolizje z budynkami."""
        route_line = LineString([(lon, lat) for lat, lon in route])
        for _, row in self.buildings_df.iterrows():
            if route_line.intersects(row.geometry):
                return True
        return False
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Bezpiecznie oblicza odległość między dwoma punktami w metrach.
        
        Args:
            point1: Pierwszy punkt (lat, lon)
            point2: Drugi punkt (lat, lon)
            
        Returns:
            float: Odległość w metrach lub 0 w przypadku błędu
        """
        try:
            if not (self._validate_coordinates(point1) and self._validate_coordinates(point2)):
                logger.warning(f"Nieprawidłowe współrzędne: {point1} lub {point2}")
                return 0
            
            # Sprawdzenie czy współrzędne nie są None lub NaN
            if any(p is None or np.isnan(p) for p in point1 + point2):
                logger.warning(f"Współrzędne zawierają None lub NaN: {point1}, {point2}")
                return 0
            
            # Konwersja punktów do układu EPSG:2180
            p1 = gpd.GeoDataFrame(
                geometry=[Point(point1[1], point1[0])],  # (lon, lat)
                crs="EPSG:4326"
            ).to_crs(epsg=2180)
            
            p2 = gpd.GeoDataFrame(
                geometry=[Point(point2[1], point2[0])],  # (lon, lat)
                crs="EPSG:4326"
            ).to_crs(epsg=2180)
            
            # Sprawdzenie czy geometrie są prawidłowe
            if p1.geometry.isna().any() or p2.geometry.isna().any():
                logger.warning("Nieprawidłowe geometrie po konwersji")
                return 0
            
            # Obliczanie odległości w metrach
            distance = float(p1.geometry[0].distance(p2.geometry[0]))
            
            # Sprawdzenie czy odległość jest prawidłowa
            if np.isnan(distance) or np.isinf(distance):
                logger.warning(f"Nieprawidłowa odległość: {distance}")
                return 0
            
            return distance
            
        except Exception as e:
            logger.warning(f"Błąd podczas obliczania odległości: {str(e)}")
            return 0

    def _is_valid_route(self, route: List[Tuple[float, float]]) -> bool:
        """Sprawdza czy trasa spełnia wszystkie ograniczenia."""
        try:
            # Sprawdzenie długości trasy
            if not (self.constraints.min_route_length <= len(route) <= self.constraints.max_route_length):
                logger.debug(f"Nieprawidłowa długość trasy: {len(route)}")
                return False
                
            # Sprawdzenie całkowitej długości
            total_length = 0
            for i in range(len(route) - 1):
                dist = self._calculate_distance(route[i], route[i + 1])
                if dist == 0:  # Błąd podczas obliczania odległości
                    return False
                total_length += dist
                
            if not (self.constraints.min_total_length <= total_length <= self.constraints.max_total_length):
                logger.debug(f"Nieprawidłowa całkowita długość trasy: {total_length}m")
                return False
                
            # Sprawdzenie początkowego przystanku
            if not self._is_valid_start_stop(route[0]):
                logger.debug("Nieprawidłowy przystanek początkowy")
                return False
                
            # Sprawdzenie odległości między przystankami i kątów
            for i in range(len(route) - 1):
                dist = self._calculate_distance(route[i], route[i + 1])
                if not (self.min_stop_distance <= dist <= self.max_stop_distance):
                    logger.debug(f"Nieprawidłowa odległość między przystankami: {dist}m")
                    return False
                    
                if i > 0:
                    angle = self._calculate_angle(route[i-1], route[i], route[i+1])
                    if angle > self.constraints.max_angle:
                        logger.debug(f"Nieprawidłowy kąt zakrętu: {angle}°")
                        return False
            
            # Sprawdzenie kolizji z istniejącymi liniami
            if self._check_collision_with_existing_lines(route):
                logger.debug("Wykryto kolizję z istniejącymi liniami")
                return False
                
            # Sprawdzenie kolizji z budynkami
            if self._check_collision_with_buildings(route):
                logger.debug("Wykryto kolizję z budynkami")
                return False
                
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
        
        # Łączna ocena
        score = (self.population_weight * density_score +
                self.distance_weight * distance_score)
                
        return score
    
    def _create_initial_population(self) -> List[List[Tuple[float, float]]]:
        """Tworzy początkową populację tras."""
        population = []
        valid_stops = [(row.geometry.y, row.geometry.x) for _, row in self.stops_df.iterrows()]
        
        logger.info(f"Liczba dostępnych przystanków: {len(valid_stops)}")
        logger.info(f"Ograniczenia: min_route_length={self.constraints.min_route_length}, "
                   f"max_route_length={self.constraints.max_route_length}")
        
        attempts = 0
        max_attempts = self.population_size * 20
        
        while len(population) < self.population_size and attempts < max_attempts:
            try:
                # Losowa długość trasy
                route_length = random.randint(
                    self.constraints.min_route_length,
                    min(self.constraints.max_route_length, len(valid_stops))
                )
                
                # Wybierz losowy punkt startowy
                start_stop = random.choice(valid_stops)
                route = [start_stop]
                
                # Dodaj pozostałe przystanki
                remaining_stops = [stop for stop in valid_stops if stop != start_stop]
                
                # Jeśli nie ma wystarczająco dużo przystanków, zmniejsz długość trasy
                if len(remaining_stops) < route_length - 1:
                    route_length = len(remaining_stops) + 1
                    logger.info(f"Dostosowano długość trasy do {route_length} (dostępne przystanki: {len(remaining_stops) + 1})")
                
                # Dodaj pozostałe przystanki
                if len(remaining_stops) > 0:
                    route.extend(random.sample(remaining_stops, route_length - 1))
                
                # Sprawdź czy trasa jest poprawna
                if self._is_valid_route(route):
                    population.append(route)
                    logger.info(f"Utworzono trasę {len(population)}/{self.population_size}")
                else:
                    logger.debug(f"Trasa nie spełnia ograniczeń: {route}")
                
            except Exception as e:
                logger.warning(f"Błąd podczas tworzenia trasy: {str(e)}")
            
            attempts += 1
            
        if len(population) == 0:
            logger.warning("Nie udało się utworzyć populacji z pełnymi ograniczeniami!")
            logger.info("Tworzę uproszczoną populację...")
            
            # Tworzymy uproszczoną populację z minimalną liczbą tras
            simplified_population = []
            for _ in range(self.population_size):
                # Wybierz dwa losowe przystanki
                if len(valid_stops) >= 2:
                    start = random.choice(valid_stops)
                    end = random.choice([s for s in valid_stops if s != start])
                    simplified_population.append([start, end])
                else:
                    # Jeśli nie ma wystarczająco dużo przystanków, użyj tego samego przystanku
                    simplified_population.append([valid_stops[0], valid_stops[0]])
            
            logger.info(f"Utworzono uproszczoną populację o rozmiarze {len(simplified_population)}")
            return simplified_population
            
        return population
    
    def _crossover(self, parent1: List[Tuple[float, float]], 
                  parent2: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], 
                                                             List[Tuple[float, float]]]:
        """Wykonuje krzyżowanie dwóch tras."""
        if random.random() > self.crossover_rate:
            return parent1, parent2
            
        point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        return child1, child2
    
    def _mutate(self, route: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Wykonuje mutację trasy."""
        if random.random() > self.mutation_rate:
            return route
            
        mutated_route = route.copy()
        mutation_type = random.choice(['swap', 'replace'])
        
        valid_stops = [(row.geometry.y, row.geometry.x) for _, row in self.stops_df.iterrows()]
        
        if mutation_type == 'swap':
            i, j = random.sample(range(len(route)), 2)
            mutated_route[i], mutated_route[j] = mutated_route[j], mutated_route[i]
        else:  # replace
            i = random.randrange(len(route))
            mutated_route[i] = random.choice(valid_stops)
            
        return mutated_route
    
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