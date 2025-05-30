import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


OPTIMIZATION_PARAMS = {
    'genetic_algorithm': {
        'population_size': 100,
        'generations': 50,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8
    },
    'weights': {
        'travel_time': 0.4,
        'coverage': 0.4,
        'cost': 0.2
    }
}


KRAKOW_BOUNDS = {
    'north': 50.1274,
    'south': 49.9736,
    'east': 20.2176,
    'west': 19.7834
}
