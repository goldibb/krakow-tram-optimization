# Ulepszenia Systemu Optymalizacji Tras Tramwajowych

## 🎯 Cel Projektu

Zaprojektowanie systemu optymalizacji tras tramwajowych w Krakowie, który:

1. **Maksymalizuje pokrycie obszarów o dużej gęstości zabudowy** - obszar w promieniu 300m od przystanków
2. **Maksymalizuje dystans między przystankami** - unika nadmiernej liczby przystanków w małych odległościach
3. **Minimalizuje liczbę zakrętów** - prostsze trasy bez nadmiaru skrzyżowań i łuków
4. **Zapewnia unikatowość przystanków** - każdy przystanek może być używany tylko w jednej trasie
5. **Gwarantuje połączenie tras** - trasy wykorzystują rzeczywiste drogi

## 🔧 Kluczowe Ulepszenia

### 1. Nowa Funkcja Celu z Trzema Kryteriami

```python
score = (population_weight * density_score + 
         distance_weight * distance_score + 
         angle_weight * angle_score)
```

**Domyślne wagi:**
- `population_weight = 0.6` - gęstość zabudowy (najważniejsze)
- `distance_weight = 0.3` - optymalne odległości między przystankami  
- `angle_weight = 0.1` - minimalizacja kątów zakrętu

### 2. Zapewnienie Unikatowości Przystanków

#### Nowe funkcje:
- `_ensure_unique_stops()` - usuwa duplikaty z tras
- `_find_alternative_stop()` - znajduje alternatywne przystanki
- `reset_used_stops()` - resetuje zestaw używanych przystanków
- `used_stops` - globalny zestaw śledzący używane przystanki

#### Mechanizm działania:
1. Każdy przystanek jest zaokrąglany do 6 miejsc po przecinku
2. System śledzi używane przystanki globalnie
3. Podczas tworzenia nowych tras wyklucza już używane przystanki
4. Oferuje alternatywne przystanki w pobliżu jeśli potrzeba

### 3. Połączenie Tras przez Rzeczywiste Drogi

#### Nowe funkcje:
- `_find_connecting_path()` - używa algorytmu A* do znajdowania ścieżek
- `_create_connected_route()` - łączy przystanki rzeczywistymi drogami
- `_extract_stops_from_route()` - wyodrębnia główne przystanki z pełnej trasy

#### Proces:
1. Przystanki są łączone przez algorytm A* na grafie ulic
2. Trasa składa się z rzeczywistych punktów na drogach
3. Główne przystanki są wyodrębniane z pełnej ścieżki

### 4. Minimalizacja Kątów Zakrętu

#### Nowa funkcja:
```python
def calculate_angle_score(self, route: List[Tuple[float, float]]) -> float:
    """Oblicza wynik dla prostoty trasy (0-1, wyższe = prostsze)"""
```

#### Algorytm:
- Oblicza kąty między kolejnymi segmentami trasy
- Kara za odchylenie od linii prostej (180°)
- Normalizacja do zakresu 0-1

### 5. Optymalizacja Wielu Tras Jednocześnie

#### Nowa funkcja:
```python
def optimize_multiple_routes(self, num_routes: int = 3) -> List[Tuple[route, score]]:
    """Optymalizuje wiele tras zapewniając unikatowość przystanków"""
```

#### Proces:
1. Optymalizuje jedną trasę
2. Oznacza jej przystanki jako używane
3. Przechodzi do następnej trasy z zaktualizowaną listą dostępnych przystanków

### 6. Ulepszone Operatory Genetyczne

#### Mutacja:
- `swap` - zamiana przystanków miejscami
- `replace` - zastąpienie przystanku nowym (unikatowym)
- `add` - dodanie nowego przystanku
- `remove` - usunięcie przystanku

#### Krzyżowanie:
- Wyodrębnia główne przystanki z tras rodziców
- Zapewnia unikatowość w potomstwie
- Tworzy połączone trasy dla potomstwa

## 📋 Zaktualizowane Parametry

### RouteConstraints:
```python
constraints = RouteConstraints(
    min_distance_between_stops=200,   # 200m (zmienione z 300m)
    max_distance_between_stops=1500,  # 1500m (zmienione z 800m)
    max_angle=60,                     # 60° (zmienione z 45°)
    min_route_length=3,               # 3 przystanki (zmienione z 5)
    max_route_length=20,              # bez zmian
    min_total_length=1000,            # 1km (zmienione z 2km)
    max_total_length=15000,           # 15km (zmienione z 10km)
    min_distance_from_buildings=3,    # 3m (zmienione z 5m)
    angle_weight=0.1                  # nowy parametr
)
```

## 🚀 Użycie

### Podstawowe użycie:
```python
from src.optimization.route_optimizer import RouteOptimizer, RouteConstraints

# Konfiguracja
constraints = RouteConstraints(...)
optimizer = RouteOptimizer(
    buildings_df=buildings_df,
    streets_df=streets_df, 
    stops_df=stops_df,
    lines_df=lines_df,
    constraints=constraints,
    population_weight=0.6,
    distance_weight=0.3,
    angle_weight=0.1
)

# Optymalizacja wielu tras
routes = optimizer.optimize_multiple_routes(num_routes=3)
```

### Test systemu:
```bash
python test_improved_optimization.py
```

### Notebook z przykładami:
```bash
jupyter notebook notebooks/route_optimization_improved.ipynb
```

## 📊 Wyniki

System zapewnia:

✅ **Unikatowość przystanków** - każdy przystanek używany tylko raz  
✅ **Połączone trasy** - wykorzystanie rzeczywistych dróg  
✅ **Minimalizację kątów** - prostsze, bardziej naturalne trasy  
✅ **Optymalizację gęstości** - pokrycie obszarów o wysokiej zabudowie  
✅ **Kontrolę odległości** - optymalne rozłożenie przystanków  

## 🔍 Struktura Plików

```
├── src/optimization/route_optimizer.py     # Główny optymalizator (ulepszony)
├── src/optimization/density_calculator.py # Kalkulator gęstości
├── src/visualization/route_visualizer.py  # Wizualizacja
├── notebooks/route_optimization_improved.ipynb # Nowy notebook
├── test_improved_optimization.py          # Testy systemu
└── README_IMPROVEMENTS.md                 # Ta dokumentacja
```

## 🎯 Kluczowe Korzyści

1. **Realność tras** - wszystkie trasy używają rzeczywistych dróg
2. **Efektywność systemu** - brak duplikowania przystanków
3. **Prostota tras** - minimalizacja skomplikowanych zakrętów  
4. **Pokrycie zaludnienia** - maksymalizacja obsługi gęsto zabudowanych obszarów
5. **Skalowalność** - możliwość optymalizacji wielu tras jednocześnie

## 🛠️ Wymagania Techniczne

- Python 3.8+
- geopandas, networkx, shapely
- numpy, matplotlib, folium
- scipy (dla spatial indexing)

System automatycznie:
- Konwertuje współrzędne między WGS84 i EPSG:2180
- Buduje graf sieci ulic z spatial indexing
- Zapewnia walidację wszystkich ograniczeń
- Oblicza szczegółowe metryki dla każdej trasy 