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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RouteConstraints:
    min_distance_between_stops: float = 300  # minimalna odległość między przystankami w metrach
    max_distance_between_stops: float = 800  # maksymalna odległość między przystankami w metrach
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
        
    def _create_street_graph(self) -> nx.Graph:
        """
        Tworzy graf sieci ulic na podstawie danych OSM.
        Inteligentnie ogranicza obszar do rejonów istotnych dla nowej linii tramwajowej.
        
        Returns:
            nx.Graph: Graf sieci ulic
        """
        G = nx.Graph()
        
        # INTELIGENTNE OGRANICZENIE OBSZARU
        logger.info("Ograniczanie obszaru do rejonów istotnych dla tramwaju...")
        
        if self.stops_df is not None and len(self.stops_df) > 0:
            # 1. Obszary wokół istniejących przystanków tramwajowych (1km bufor)
            stops_buffer_distance = 1000  # 1km w metrach
            stops_buffers = self.stops_projected.geometry.buffer(stops_buffer_distance)
            stops_union = unary_union(stops_buffers)
            logger.info(f"Utworzono bufor 1km wokół {len(self.stops_df)} przystanków")
            
            # 2. Obszary o wysokiej gęstości zabudowy
            if len(self.buildings_projected) > 0:
                # Oblicz gęstość zabudowy w siatce 500x500m
                buildings_bounds = self.buildings_projected.total_bounds
                grid_size = 500  # 500m siatka
                
                high_density_areas = []
                min_x, min_y, max_x, max_y = buildings_bounds
                
                for x in range(int(min_x), int(max_x), grid_size):
                    for y in range(int(min_y), int(max_y), grid_size):
                        # Prostokąt siatki
                        from shapely.geometry import box
                        grid_cell = box(x, y, x + grid_size, y + grid_size)
                        
                        # Znajdź budynki w tym obszarze
                        buildings_in_cell = self.buildings_projected[
                            self.buildings_projected.geometry.intersects(grid_cell)
                        ]
                        
                        # Jeśli gęstość > próg, dodaj do obszarów zainteresowania
                        if len(buildings_in_cell) > 10:  # próg: min 10 budynków na 500x500m
                            high_density_areas.append(grid_cell)
                
                if high_density_areas:
                    density_union = unary_union(high_density_areas)
                    # Połącz obszary przystanków i gęstej zabudowy
                    relevant_area = unary_union([stops_union, density_union])
                    logger.info(f"Znaleziono {len(high_density_areas)} obszarów wysokiej gęstości zabudowy")
                else:
                    relevant_area = stops_union
                    logger.info("Brak obszarów wysokiej gęstości - używam tylko buforów przystanków")
            else:
                relevant_area = stops_union
                logger.info("Brak danych o budynkach - używam tylko buforów przystanków")
            
            # 3. Filtruj ulice do istotnych obszarów
            logger.info("Filtrowanie ulic do istotnych obszarów...")
            streets_in_relevant_area = self.streets_projected[
                self.streets_projected.geometry.intersects(relevant_area)
            ]
            
            logger.info(f"Ograniczono z {len(self.streets_projected)} do {len(streets_in_relevant_area)} ulic")
            
            # Jeśli nadal za dużo, weź próbkę
            if len(streets_in_relevant_area) > 5000:
                streets_filtered = streets_in_relevant_area.sample(n=5000, random_state=42)
                logger.info(f"Dodatkowo ograniczono do {len(streets_filtered)} ulic (próbka)")
            else:
                streets_filtered = streets_in_relevant_area
                
        else:
            # Fallback - brak przystanków
            logger.warning("Brak danych o przystankach - używam ograniczonej próbki ulic")
            streets_filtered = self.streets_projected.sample(n=min(2000, len(self.streets_projected)), random_state=42)
        
        # Sprawdź czy mamy jakieś ulice
        if len(streets_filtered) == 0:
            logger.error("Brak ulic po filtrowaniu! Używam próbki 1000 ulic.")
            streets_filtered = self.streets_projected.head(1000)
        
        logger.info(f"Finalna liczba ulic do przetworzenia: {len(streets_filtered)}")
        
        # Dodawanie węzłów (skrzyżowania)
        for idx, row in streets_filtered.iterrows():
            coords = list(row.geometry.coords)
            for i in range(len(coords) - 1):
                # coords są już w układzie EPSG:2180 (x, y)
                point1_epsg2180 = coords[i]  # (x, y) w EPSG:2180
                point2_epsg2180 = coords[i + 1]  # (x, y) w EPSG:2180
                
                # Obliczanie odległości używając bezpiecznej metody z EPSG:2180
                dist = self._calculate_distance(point1_epsg2180, point2_epsg2180, is_wgs84=False)
                
                # Dodaj krawędź tylko jeśli odległość jest prawidłowa
                if dist > 0:
                    G.add_edge(
                        point1_epsg2180,
                        point2_epsg2180,
                        weight=dist
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
    
    def _find_nearest_point_in_graph(self, point: Tuple[float, float], max_distance: float = 100) -> Optional[Tuple[float, float]]:
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
        start_point_in_graph = self._find_nearest_point_in_graph(start_point)
        end_point_in_graph = self._find_nearest_point_in_graph(end_point)
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
        
        # Łączna ocena
        score = (self.population_weight * density_score +
                self.distance_weight * distance_score)
                
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
                if self._is_valid_route(route, is_simplified=False):
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
                try:
                    # Wybierz dwa losowe przystanki
                    if len(valid_stops) >= 2:
                        start = random.choice(valid_stops)
                        end = random.choice([s for s in valid_stops if s != start])
                        route = [start, end]
                        
                        # Sprawdź czy trasa jest poprawna
                        if self._is_valid_route(route, is_simplified=True):
                            simplified_population.append(route)
                        else:
                            # Jeśli trasa nie jest poprawna, dodaj ją mimo to
                            simplified_population.append(route)
                    else:
                        # Jeśli nie ma wystarczająco dużo przystanków, użyj tego samego przystanku
                        simplified_population.append([valid_stops[0], valid_stops[0]])
                except Exception as e:
                    logger.warning(f"Błąd podczas tworzenia uproszczonej trasy: {str(e)}")
                    continue
            
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
        
        # Używamy oryginalnego stops_df w WGS84, nie projected
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