import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Tuple, Dict, Optional
import random
from dataclasses import dataclass
import logging
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import cKDTree
import networkx as nx
import time
from .density_calculator import DensityCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RouteConstraints:
    """Ograniczenia dla optymalizacji tras tramwajowych."""
    min_distance_between_stops: float = 350    # minimalna odległość między przystankami [m]
    max_distance_between_stops: float = 700    # maksymalna odległość między przystankami [m]
    min_total_length: float = 1500             # minimalna długość trasy [m]
    max_total_length: float = 15000            # maksymalna długość trasy [m]
    min_route_stops: int = 4                   # minimalna liczba przystanków
    max_route_stops: int = 15                  # maksymalna liczba przystanków
    min_distance_from_buildings: float = 5.0   # minimalna odległość od budynków [m]
    buffer_around_existing_lines: float = 50.0 # bufor wokół istniejących linii [m]

class SmartRouteOptimizer:
    """
    Inteligentny optymalizator tras tramwajowych z uczeniem się.
    
    Algorytm:
    1. Buduje trasy lokalnie (bez "skakania")
    2. Unika kolizji z budynkami i istniejącymi liniami
    3. Maksymalizuje gęstość zabudowy w promieniu 300m
    4. Optymalizuje odległości między przystankami
    5. Uczy się z poprzednich iteracji
    """
    
    def __init__(self, 
                 buildings_df: gpd.GeoDataFrame,
                 streets_df: gpd.GeoDataFrame,
                 stops_df: gpd.GeoDataFrame,
                 lines_df: Optional[gpd.GeoDataFrame] = None,
                 constraints: Optional[RouteConstraints] = None):
        """
        Inicjalizacja optymalizatora.
        
        Args:
            buildings_df: DataFrame z budynkami
            streets_df: DataFrame z ulicami
            stops_df: DataFrame z przystankami
            lines_df: DataFrame z istniejącymi liniami tramwajowymi
            constraints: Ograniczenia optymalizacji
        """
        self.buildings_df = buildings_df
        self.streets_df = streets_df
        self.stops_df = stops_df
        self.lines_df = lines_df
        self.constraints = constraints or RouteConstraints()
        
        # Konwersja do układu metrycznego (EPSG:2180 dla Polski)
        self.buildings_projected = buildings_df.to_crs(epsg=2180)
        self.streets_projected = streets_df.to_crs(epsg=2180)
        self.stops_projected = stops_df.to_crs(epsg=2180)
        if lines_df is not None:
            self.lines_projected = lines_df.to_crs(epsg=2180)
        
        # Inicjalizacja komponentów
        self._initialize_density_calculator()
        self._create_buildings_buffer()
        self._create_existing_lines_buffer()
        self._create_water_buffer()
        self._create_stops_kdtree()
        
        # Pamięć algorytmu - uczenie się
        self.successful_routes_memory = []
        self.good_connections_memory = {}
        self.bad_areas_memory = set()
        
        logger.info("SmartRouteOptimizer zainicjalizowany pomyślnie")
    
    def _initialize_density_calculator(self):
        """Inicjalizuje kalkulator gęstości zabudowy."""
        self.density_calculator = DensityCalculator(
            self.buildings_df, 
            radius_meters=300
        )
        logger.info("Kalkulator gęstości zainicjalizowany")
    
    def _create_buildings_buffer(self):
        """Tworzy bufor bezpieczeństwa wokół budynków."""
        try:
            buildings_union = unary_union(self.buildings_projected.geometry)
            self.buildings_buffer = buildings_union.buffer(
                self.constraints.min_distance_from_buildings
            )
            logger.info(f"Utworzono bufor {self.constraints.min_distance_from_buildings}m wokół budynków")
        except Exception as e:
            logger.warning(f"Błąd tworzenia buforu budynków: {e}")
            self.buildings_buffer = None
    
    def _create_existing_lines_buffer(self):
        """Tworzy bufor wokół istniejących linii tramwajowych."""
        self.existing_lines_buffer = None
        if self.lines_df is not None and len(self.lines_df) > 0:
            try:
                lines_union = unary_union(self.lines_projected.geometry)
                self.existing_lines_buffer = lines_union.buffer(
                    self.constraints.buffer_around_existing_lines
                )
                logger.info(f"Utworzono bufor {self.constraints.buffer_around_existing_lines}m wokół istniejących linii")
            except Exception as e:
                logger.warning(f"Błąd tworzenia buforu linii: {e}")
    
    def _create_water_buffer(self):
        """Tworzy bufor wokół obszarów wodnych na podstawie danych OpenStreetMap."""
        self.water_buffer = None
        try:
            # Spróbuj znaleźć obszary wodne w danych streets (często tam są kodowane)
            water_features = []
            
            # Sprawdź czy są kolumny związane z wodą
            water_columns = ['waterway', 'natural', 'landuse', 'water', 'highway']
            
            for col in water_columns:
                if col in self.streets_df.columns:
                    # Filtruj obiekty wodne
                    water_mask = self.streets_df[col].isin([
                        'river', 'stream', 'canal', 'water', 'riverbank', 
                        'waterway', 'wetland', 'lake', 'pond'
                    ])
                    if water_mask.any():
                        water_features.extend(self.streets_projected[water_mask].geometry.tolist())
            
            # Dodaj duże obszary na podstawie geometrii (Wisła ma dużą szerokość)
            for geom in self.streets_projected.geometry:
                try:
                    # Jeśli geometria ma dużą powierzchnię, może być wodą
                    if hasattr(geom, 'area') and geom.area > 50000:  # 5 hektarów
                        water_features.append(geom)
                except:
                    continue
            
            if water_features:
                water_union = unary_union(water_features)
                self.water_buffer = water_union.buffer(50)  # 50m bufor wokół wody
                logger.info(f"Utworzono bufor 50m wokół {len(water_features)} obiektów wodnych")
            else:
                # Jako fallback - utwórz przybliżony bufor Wisły na podstawie znanej lokalizacji
                from shapely.geometry import box
                # Wisła w Krakowie - przybliżone współrzędne
                wisla_bbox = box(19.88, 50.04, 19.98, 50.08)  # WGS84
                wisla_gdf = gpd.GeoDataFrame(geometry=[wisla_bbox], crs="EPSG:4326").to_crs(epsg=2180)
                self.water_buffer = wisla_gdf.geometry[0].buffer(100)
                logger.info("Utworzono przybliżony bufor Wisły (100m)")
                
        except Exception as e:
            logger.warning(f"Błąd tworzenia buforu wody: {e}")
            self.water_buffer = None
    
    def _create_stops_kdtree(self):
        """Tworzy KDTree dla szybkiego wyszukiwania najbliższych przystanków."""
        try:
            stops_coords = np.array([
                [geom.x, geom.y] for geom in self.stops_projected.geometry
            ])
            self.stops_kdtree = cKDTree(stops_coords)
            self.stops_coords_array = stops_coords
            logger.info(f"KDTree utworzone dla {len(stops_coords)} przystanków")
        except Exception as e:
            logger.error(f"Błąd tworzenia KDTree: {e}")
            self.stops_kdtree = None
    
    def find_high_density_stops(self, top_n: int = 20) -> List[Tuple[float, float]]:
        """
        Znajduje przystanki o najwyższej gęstości zabudowy.
        
        Args:
            top_n: Liczba przystanków do zwrócenia
            
        Returns:
            Lista współrzędnych (lat, lon) najlepszych przystanków
        """
        stop_densities = []
        
        logger.info(f"Analizuję gęstość zabudowy dla {len(self.stops_df)} przystanków...")
        
        for idx, stop in self.stops_df.iterrows():
            try:
                # Oblicz gęstość zabudowy w promieniu 300m
                density = self.density_calculator.calculate_density_at_point(
                    stop.geometry.y, stop.geometry.x
                )
                
                stop_densities.append({
                    'coords': (stop.geometry.y, stop.geometry.x),
                    'density': density,
                    'index': idx
                })
            except Exception as e:
                logger.debug(f"Błąd obliczania gęstości dla przystanku {idx}: {e}")
                continue
        
        # Sortuj według gęstości
        stop_densities.sort(key=lambda x: x['density'], reverse=True)
        
        # Loguj wyniki
        logger.info("TOP 5 przystanków według gęstości zabudowy:")
        for i, stop in enumerate(stop_densities[:5]):
            logger.info(f"  {i+1}. Gęstość: {stop['density']:.2f}, Coords: {stop['coords']}")
        
        return [stop['coords'] for stop in stop_densities[:top_n]]
    
    def find_nearby_stops(self, center_lat: float, center_lon: float, 
                         max_distance: float = 1000) -> List[Tuple[float, float, float]]:
        """
        Znajduje przystanki w pobliżu danego punktu.
        
        Args:
            center_lat: Szerokość geograficzna centrum
            center_lon: Długość geograficzna centrum
            max_distance: Maksymalna odległość w metrach
            
        Returns:
            Lista tupli (lat, lon, distance)
        """
        if self.stops_kdtree is None:
            return []
        
        try:
            # Konwertuj punkt centrum do EPSG:2180
            center_gdf = gpd.GeoDataFrame(
                geometry=[Point(center_lon, center_lat)], 
                crs="EPSG:4326"
            ).to_crs(epsg=2180)
            center_projected = center_gdf.geometry[0]
            center_coords = np.array([center_projected.x, center_projected.y])
            
            # Znajdź najbliższe przystanki
            distances, indices = self.stops_kdtree.query(
                center_coords, 
                k=min(50, len(self.stops_coords_array)),
                distance_upper_bound=max_distance
            )
            
            nearby_stops = []
            for dist, idx in zip(distances, indices):
                if dist < np.inf and dist <= max_distance:
                    # Konwertuj z powrotem do WGS84
                    stop_projected = Point(self.stops_coords_array[idx])
                    stop_gdf = gpd.GeoDataFrame(
                        geometry=[stop_projected], 
                        crs="EPSG:2180"
                    ).to_crs(epsg=4326)
                    stop_wgs84 = stop_gdf.geometry[0]
                    
                    nearby_stops.append((stop_wgs84.y, stop_wgs84.x, dist))
            
            return nearby_stops
            
        except Exception as e:
            logger.error(f"Błąd wyszukiwania pobliskich przystanków: {e}")
            return []
    
    def is_route_safe(self, route: List[Tuple[float, float]]) -> Tuple[bool, str]:
        """
        Sprawdza czy trasa jest bezpieczna (nie przecina budynków, wody ani istniejących linii).
        
        Args:
            route: Lista punktów trasy (lat, lon)
            
        Returns:
            Tuple (czy_bezpieczna, opis_problemów)
        """
        if len(route) < 2:
            return True, "Trasa za krótka do sprawdzenia"
        
        try:
            # Konwertuj trasę do EPSG:2180
            route_points_projected = []
            for lat, lon in route:
                point_gdf = gpd.GeoDataFrame(
                    geometry=[Point(lon, lat)], 
                    crs="EPSG:4326"
                ).to_crs(epsg=2180)
                route_points_projected.append((point_gdf.geometry[0].x, point_gdf.geometry[0].y))
            
            route_line = LineString(route_points_projected)
            
            # Sprawdź czy trasa nie jest zbyt długa (może przechodzić przez wodę)
            route_length = route_line.length
            if route_length > self.constraints.max_distance_between_stops * 2:
                return False, f"Segment za długi: {route_length:.0f}m"
            
            # Sprawdź kolizje z budynkami - bardziej restrykcyjne
            if self.buildings_buffer is not None:
                buffer_intersection = route_line.intersection(self.buildings_buffer)
                if not buffer_intersection.is_empty and buffer_intersection.length > 10:  # Bardzo mała tolerancja
                    return False, "Kolizja z budynkami"
            
            # Sprawdź kolizje z istniejącymi liniami - bardziej restrykcyjne  
            if self.existing_lines_buffer is not None:
                lines_intersection = route_line.intersection(self.existing_lines_buffer)
                if not lines_intersection.is_empty and lines_intersection.length > 20:  # Mała tolerancja
                    return False, "Kolizja z istniejącymi liniami tramwajowymi"
            
            # Sprawdź kolizje z wodą - BRAK TOLERANCJI
            if self.water_buffer is not None:
                water_intersection = route_line.intersection(self.water_buffer)
                if not water_intersection.is_empty:
                    return False, "Kolizja z wodą/rzeką"
            
            # Dodatkowe sprawdzenie - czy trasa nie ma dziwnych skoków
            for i in range(len(route) - 1):
                segment_distance = self._calculate_distance_wgs84(route[i], route[i + 1])
                if segment_distance > self.constraints.max_distance_between_stops * 2:
                    return False, f"Segment zbyt długi: {segment_distance:.0f}m (możliwa kolizja z wodą)"
            
            return True, "Trasa bezpieczna"
            
        except Exception as e:
            logger.debug(f"Błąd sprawdzania bezpieczeństwa: {e}")
            return False, f"Błąd sprawdzania bezpieczeństwa: {e}"
    
    def calculate_route_score(self, route: List[Tuple[float, float]]) -> float:
        """
        Oblicza ocenę trasy na podstawie gęstości zabudowy i odległości między przystankami.
        
        Args:
            route: Lista punktów trasy (lat, lon)
            
        Returns:
            Ocena trasy (0-100)
        """
        if len(route) < 2:
            return 0.0
        
        try:
            # 1. Ocena gęstości zabudowy (waga 60%)
            density_score = 0.0
            for lat, lon in route:
                density = self.density_calculator.calculate_density_at_point(lat, lon)
                density_score += density
            density_score = (density_score / len(route)) * 0.6
            
            # 2. Ocena odległości między przystankami (waga 40%)
            distance_score = 0.0
            valid_segments = 0
            
            for i in range(len(route) - 1):
                distance = self._calculate_distance_wgs84(route[i], route[i + 1])
                
                # Punkty za optymalne odległości
                if self.constraints.min_distance_between_stops <= distance <= self.constraints.max_distance_between_stops:
                    # Im bliżej środka zakresu, tym lepiej
                    optimal_distance = (self.constraints.min_distance_between_stops + 
                                      self.constraints.max_distance_between_stops) / 2
                    distance_penalty = abs(distance - optimal_distance) / optimal_distance
                    segment_score = max(0, 1 - distance_penalty)
                    distance_score += segment_score
                    valid_segments += 1
            
            if valid_segments > 0:
                distance_score = (distance_score / valid_segments) * 0.4
            
            # 3. Bonus za uczenie się
            learning_bonus = self._calculate_learning_bonus(route)
            
            total_score = (density_score + distance_score + learning_bonus) * 100
            return min(100, max(0, total_score))
            
        except Exception as e:
            logger.error(f"Błąd obliczania oceny trasy: {e}")
            return 0.0
    
    def _calculate_distance_wgs84(self, point1: Tuple[float, float], 
                                 point2: Tuple[float, float]) -> float:
        """Oblicza odległość między dwoma punktami w WGS84."""
        try:
            # Konwertuj do EPSG:2180 dla precyzyjnych obliczeń
            p1_gdf = gpd.GeoDataFrame(
                geometry=[Point(point1[1], point1[0])], 
                crs="EPSG:4326"
            ).to_crs(epsg=2180)
            
            p2_gdf = gpd.GeoDataFrame(
                geometry=[Point(point2[1], point2[0])], 
                crs="EPSG:4326"
            ).to_crs(epsg=2180)
            
            return p1_gdf.geometry[0].distance(p2_gdf.geometry[0])
            
        except Exception as e:
            logger.error(f"Błąd obliczania odległości: {e}")
            return 0.0
    
    def _calculate_learning_bonus(self, route: List[Tuple[float, float]]) -> float:
        """Oblicza bonus na podstawie pamięci algorytmu."""
        bonus = 0.0
        
        # Bonus za wykorzystanie sprawdzonych połączeń
        for i in range(len(route) - 1):
            connection = (route[i], route[i + 1])
            if connection in self.good_connections_memory:
                bonus += 0.02 * self.good_connections_memory[connection]
        
        # Kara za obszary które się nie sprawdziły
        for point in route:
            rounded_point = (round(point[0], 4), round(point[1], 4))
            if rounded_point in self.bad_areas_memory:
                bonus -= 0.01
        
        return bonus
    
    def build_local_route(self, start_lat: float, start_lon: float, 
                         target_stops: int = 8) -> List[Tuple[float, float]]:
        """
        Buduje trasę lokalnie, krok po kroku, wybierając najbliższe przystanki.
        
        Args:
            start_lat: Szerokość geograficzna punktu startowego
            start_lon: Długość geograficzna punktu startowego
            target_stops: Docelowa liczba przystanków
            
        Returns:
            Lista punktów trasy (lat, lon)
        """
        route = [(start_lat, start_lon)]
        used_stops = {(round(start_lat, 6), round(start_lon, 6))}
        
        current_lat, current_lon = start_lat, start_lon
        
        logger.debug(f"Budowanie lokalnej trasy z {target_stops} przystankami...")
        
        for step in range(target_stops - 1):
            # Znajdź pobliskie przystanki - bardziej konserwatywny zasięg
            search_radius = self.constraints.max_distance_between_stops * (1.2 + step * 0.1)  # Mniejszy zasięg
            nearby_stops = self.find_nearby_stops(
                current_lat, current_lon, 
                max_distance=search_radius
            )
            
            # Filtruj już używane przystanki - bardziej restrykcyjne odległości
            min_distance = self.constraints.min_distance_between_stops * 0.8  # Zwiększona minimalna odległość
            available_stops = [
                (lat, lon, dist) for lat, lon, dist in nearby_stops
                if (round(lat, 6), round(lon, 6)) not in used_stops
                and dist >= min_distance
                and dist <= self.constraints.max_distance_between_stops * 1.5  # Maksymalna odległość
            ]
            
            if not available_stops:
                logger.debug(f"Brak dostępnych przystanków w kroku {step + 1}, zasięg: {search_radius:.0f}m")
                break
            
            # Wybierz najlepszy przystanek na podstawie gęstości i odległości
            best_stop = None
            best_score = -1
            
            # ZAWSZE sprawdzaj bezpieczeństwo - usunąłem pomijanie dla pierwszych kroków
            for lat, lon, dist in available_stops[:10]:  # Mniej kandydatów dla szybkości
                
                # ZAWSZE sprawdzaj bezpieczeństwo połączenia
                test_route = route + [(lat, lon)]
                is_safe, safety_reason = self.is_route_safe(test_route)
                
                if is_safe:
                    # Oblicz ocenę przystanku
                    density = self.density_calculator.calculate_density_at_point(lat, lon)
                    
                    # Sprawdź czy odległość jest w dozwolonym zakresie
                    if self.constraints.min_distance_between_stops <= dist <= self.constraints.max_distance_between_stops:
                        # Ocena: gęstość (70%) + odległość (30%)
                        distance_score = 1.0 - abs(dist - 525) / 525  # Preferuj ~525m (środek zakresu)
                        combined_score = density * 0.7 + distance_score * 0.3
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_stop = (lat, lon)
                else:
                    logger.debug(f"Odrzucono przystanek ({lat:.6f}, {lon:.6f}) - {safety_reason}")
            
            if best_stop is None:
                logger.debug(f"Nie znaleziono bezpiecznego przystanku w kroku {step + 1}")
                break
            else:
                route.append(best_stop)
                used_stops.add((round(best_stop[0], 6), round(best_stop[1], 6)))
                current_lat, current_lon = best_stop
                logger.debug(f"Dodano przystanek {len(route)}: ({best_stop[0]:.6f}, {best_stop[1]:.6f}), ocena: {best_score:.3f}")
        
        logger.debug(f"Zbudowano lokalną trasę z {len(route)} przystankami")
        return route
    
    def optimize_routes(self, num_routes: int = 3, 
                       max_iterations: int = 100) -> List[Tuple[List[Tuple[float, float]], float]]:
        """
        Główna metoda optymalizacji tras tramwajowych.
        
        Args:
            num_routes: Liczba tras do zoptymalizowania
            max_iterations: Maksymalna liczba iteracji na trasę
            
        Returns:
            Lista tupli (trasa, ocena) posortowana według oceny
        """
        logger.info(f"Rozpoczynam optymalizację {num_routes} tras...")
        
        # Znajdź przystanki o wysokiej gęstości zabudowy
        high_density_stops = self.find_high_density_stops(top_n=min(50, len(self.stops_df)))
        
        logger.info(f"TOP 5 przystanków według gęstości zabudowy:")
        for i, (lat, lon) in enumerate(high_density_stops[:5]):
            density = self.density_calculator.calculate_density_at_point(lat, lon)
            logger.info(f"  {i+1}. Gęstość: {density:.2f}, Coords: ({lat}, {lon})")
        
        optimized_routes = []
        
        for route_num in range(num_routes):
            logger.info(f"Optymalizacja trasy {route_num + 1}/{num_routes}")
            
            best_route = None
            best_score = -1
            valid_routes_found = 0
            
            # Próbuj różne punkty startowe
            attempts = min(max_iterations, len(high_density_stops))
            logger.info(f"Rozpoczynam {attempts} prób dla trasy {route_num + 1}")
            
            for attempt in range(attempts):
                try:
                    # Loguj postęp co 10 prób
                    if attempt % 10 == 0 or attempt == attempts - 1:
                        logger.info(f"🔄 Trasa {route_num + 1}: Próba {attempt + 1}/{attempts} ({(attempt + 1) / attempts * 100:.0f}%)")
                    
                    # Wybierz losowy punkt startowy z wysoką gęstością
                    start_idx = attempt % len(high_density_stops)
                    start_lat, start_lon = high_density_stops[start_idx]
                    
                    # Zbuduj trasę lokalnie - celuj w większą liczbę przystanków
                    target_stops = np.random.randint(
                        self.constraints.min_route_stops, 
                        self.constraints.max_route_stops + 1
                    )
                    
                    route = self.build_local_route(start_lat, start_lon, target_stops)
                    
                    # WAŻNE: Odrzuć trasy z mniej niż minimalną liczbą przystanków
                    if len(route) >= self.constraints.min_route_stops:
                        valid_routes_found += 1
                        
                        # Oblicz ocenę trasy
                        score = self.calculate_route_score(route)
                        
                        logger.debug(f"Próba {attempt + 1}: {len(route)} przystanków, ocena: {score:.2f}")
                        
                        if score > best_score:
                            best_route = route
                            best_score = score
                            
                            # Zapisz w pamięci algorytmu
                            self._learn_from_route(route, score)
                            
                            logger.info(f"⭐ Nowa najlepsza trasa {route_num + 1}: {len(route)} przystanków, ocena: {score:.2f} [próba {attempt + 1}/{attempts}]")
                    else:
                        logger.debug(f"Próba {attempt + 1}: Odrzucono trasę z {len(route)} przystankami (min: {self.constraints.min_route_stops})")
                            
                except Exception as e:
                    logger.debug(f"Błąd w próbie {attempt + 1}: {e}")
                    continue
            
            if best_route is not None:
                optimized_routes.append((best_route, best_score))
                logger.info(f"✅ Znaleziono trasę {route_num + 1}: {len(best_route)} przystanków, ocena: {best_score:.2f} (sprawdzono {valid_routes_found} prawidłowych tras)")
            else:
                logger.warning(f"❌ Nie udało się zoptymalizować trasy {route_num + 1} (sprawdzono {valid_routes_found} prawidłowych tras)")
        
        # Sortuj według oceny
        optimized_routes.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Optymalizacja zakończona: {len(optimized_routes)}/{num_routes} tras")
        return optimized_routes
    
    def get_optimization_stats(self) -> Dict:
        """Zwraca statystyki optymalizacji."""
        return {
            'successful_routes': len(self.successful_routes_memory),
            'learned_connections': len(self.good_connections_memory),
            'bad_areas': len(self.bad_areas_memory),
            'total_stops': len(self.stops_df),
            'total_buildings': len(self.buildings_df)
        }
    
    def _learn_from_route(self, route: List[Tuple[float, float]], score: float):
        """
        Uczy się z trasą - zapisuje ją w pamięci algorytmu.
        
        Args:
            route: Lista punktów trasy (lat, lon)
            score: Ocena trasy
        """
        try:
            # Dodaj trasę do pamięci udanych tras
            self.successful_routes_memory.append(route)
            
            # Zapisz dobre połączenia między przystankami
            for i in range(len(route) - 1):
                connection = (route[i], route[i + 1])
                self.good_connections_memory[connection] = self.good_connections_memory.get(connection, 0) + 1
            
            # Jeśli trasa ma niską ocenę, dodaj punkty do złych obszarów
            if score < 30:
                for point in route:
                    rounded_point = (round(point[0], 4), round(point[1], 4))
                    self.bad_areas_memory.add(rounded_point)
                    
        except Exception as e:
            logger.debug(f"Błąd podczas uczenia się z trasy: {e}") 