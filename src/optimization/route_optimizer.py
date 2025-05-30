import numpy as np
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass
import logging
from .density_calculator import DensityCalculator
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
import networkx as nx
from shapely.ops import unary_union

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
    def __init__(
        self,
        buildings_df: gpd.GeoDataFrame,
        streets_df: gpd.GeoDataFrame,
        stops_df: gpd.GeoDataFrame,
        lines_df: gpd.GeoDataFrame,
        constraints: RouteConstraints = None,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8
    ):
        """
        Inicjalizacja optymalizatora trasy.
        
        Args:
            buildings_df: DataFrame z budynkami
            streets_df: DataFrame z ulicami
            stops_df: DataFrame z istniejącymi przystankami
            lines_df: DataFrame z istniejącymi liniami tramwajowymi
            constraints: Ograniczenia dla trasy
            population_size: Rozmiar populacji
            generations: Liczba pokoleń
            mutation_rate: Współczynnik mutacji
            crossover_rate: Współczynnik krzyżowania
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
        
        self.density_calculator = DensityCalculator(buildings_df)
        self.street_graph = self._create_street_graph()
        self.existing_lines = self._prepare_existing_lines()
        self.buildings_buffer = self._create_buildings_buffer()
        
    def _create_street_graph(self) -> nx.Graph:
        """Tworzy graf ulic do sprawdzania możliwości połączenia przystanków."""
        G = nx.Graph()
        
        for idx, row in self.streets_df.iterrows():
            if isinstance(row.geometry, LineString):
                coords = list(row.geometry.coords)
                for i in range(len(coords) - 1):
                    G.add_edge(coords[i], coords[i+1], weight=self._calculate_edge_weight(coords[i], coords[i+1]))
        
        return G
    
    def _calculate_edge_weight(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Oblicza wagę krawędzi na podstawie odległości i kąta."""
        distance = self._calculate_distance(point1, point2)
        return distance
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Oblicza odległość między dwoma punktami w metrach."""
        return Point(point1).distance(Point(point2)) * 111000  # konwersja stopni na metry
    
    def _calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """Oblicza kąt między trzema punktami."""
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _prepare_existing_lines(self) -> List[LineString]:
        """Przygotowuje geometrie istniejących linii tramwajowych."""
        existing_lines = []
        for _, row in self.lines_df.iterrows():
            if isinstance(row.geometry, LineString):
                existing_lines.append(row.geometry)
        return existing_lines
    
    def _create_buildings_buffer(self) -> Polygon:
        """Tworzy bufor wokół budynków."""
        buildings_union = unary_union(self.buildings_df.geometry)
        return buildings_union.buffer(self.constraints.min_distance_from_buildings / 111000)  # konwersja na stopnie
    
    def _calculate_total_length(self, route: List[Tuple[float, float]]) -> float:
        """Oblicza całkowitą długość trasy w metrach."""
        total_length = 0
        for i in range(len(route) - 1):
            total_length += self._calculate_distance(route[i], route[i+1])
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
        for existing_line in self.existing_lines:
            if route_line.distance(existing_line) < 0.0001:  # mała tolerancja
                return True
        return False
    
    def _check_collision_with_buildings(self, route: List[Tuple[float, float]]) -> bool:
        """Sprawdza kolizje z budynkami."""
        route_line = LineString([(lon, lat) for lat, lon in route])
        return route_line.intersects(self.buildings_buffer)
    
    def _is_valid_route(self, route: List[Tuple[float, float]]) -> bool:
        """Sprawdza czy trasa spełnia wszystkie ograniczenia."""
        # Sprawdzenie długości trasy
        if not (self.constraints.min_route_length <= len(route) <= self.constraints.max_route_length):
            return False
            
        # Sprawdzenie całkowitej długości
        total_length = self._calculate_total_length(route)
        if not (self.constraints.min_total_length <= total_length <= self.constraints.max_total_length):
            return False
            
        # Sprawdzenie początkowego przystanku
        if not self._is_valid_start_stop(route[0]):
            return False
            
        # Sprawdzenie odległości między przystankami i kątów
        for i in range(len(route) - 1):
            distance = self._calculate_distance(route[i], route[i+1])
            if not (self.constraints.min_distance_between_stops <= distance <= self.constraints.max_distance_between_stops):
                return False
                
            if i > 0:
                angle = self._calculate_angle(route[i-1], route[i], route[i+1])
                if angle > self.constraints.max_angle:
                    return False
        
        # Sprawdzenie kolizji z istniejącymi liniami
        if self._check_collision_with_existing_lines(route):
            return False
            
        # Sprawdzenie kolizji z budynkami
        if self._check_collision_with_buildings(route):
            return False
            
        return True
    
    def _evaluate_route(self, route: List[Tuple[float, float]]) -> float:
        """Ocenia jakość trasy."""
        if not self._is_valid_route(route):
            return float('-inf')
            
        # Obliczanie gęstości zabudowy
        density_score = self.density_calculator.calculate_density_for_route(route)
        
        # Obliczanie średniej odległości między przystankami
        distances = [self._calculate_distance(route[i], route[i+1]) for i in range(len(route)-1)]
        distance_score = np.mean(distances)
        
        # Obliczanie liczby zakrętów
        angles = [self._calculate_angle(route[i-1], route[i], route[i+1]) 
                 for i in range(1, len(route)-1)]
        turn_penalty = sum(angle for angle in angles if angle > 30)
        
        # Łączna ocena
        score = (density_score * 0.5 +  # waga dla gęstości zabudowy
                distance_score * 0.3 +  # waga dla odległości między przystankami
                -turn_penalty * 0.2)    # waga dla kary za zakręty
                
        return score
    
    def _create_initial_population(self) -> List[List[Tuple[float, float]]]:
        """Tworzy początkową populację tras."""
        population = []
        valid_stops = [(row.geometry.y, row.geometry.x) for _, row in self.stops_df.iterrows()]
        
        for _ in range(self.population_size):
            route_length = random.randint(self.constraints.min_route_length, 
                                        self.constraints.max_route_length)
            
            # Zawsze zaczynamy od istniejącego przystanku
            start_stop = random.choice(valid_stops)
            route = [start_stop]
            
            # Dodajemy pozostałe przystanki
            remaining_stops = [stop for stop in valid_stops if stop != start_stop]
            route.extend(random.sample(remaining_stops, route_length - 1))
            
            if self._is_valid_route(route):
                population.append(route)
                
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
        mutation_type = random.choice(['swap', 'replace', 'insert'])
        
        if mutation_type == 'swap':
            i, j = random.sample(range(len(route)), 2)
            mutated_route[i], mutated_route[j] = mutated_route[j], mutated_route[i]
        elif mutation_type == 'replace':
            i = random.randrange(len(route))
            mutated_route[i] = random.choice(self.potential_stops)
        else:  # insert
            i = random.randrange(len(route))
            mutated_route.insert(i, random.choice(self.potential_stops))
            
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