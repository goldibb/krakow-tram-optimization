import geopandas as gpd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_file(file_path):
    logger.info(f"Analyzing {file_path}...")
    gdf = gpd.read_file(file_path)
    
    # Convert to EPSG:2180 for analysis
    gdf_projected = gdf.to_crs(epsg=2180)
    
    # Get bounds of all geometries
    bounds = gdf_projected.total_bounds
    
    logger.info(f"X range: {bounds[0]:.2f} to {bounds[2]:.2f}")
    logger.info(f"Y range: {bounds[1]:.2f} to {bounds[3]:.2f}")
    
    return bounds

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    files = ['buildings.geojson', 'streets.geojson', 'stops.geojson', 'lines.geojson']
    
    all_bounds = []
    for file in files:
        file_path = os.path.join(data_dir, file)
        bounds = analyze_file(file_path)
        all_bounds.append(bounds)
    
    # Calculate overall bounds
    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)
    
    logger.info("\nOverall coordinate ranges:")
    logger.info(f"X range: {min_x:.2f} to {max_x:.2f}")
    logger.info(f"Y range: {min_y:.2f} to {max_y:.2f}")

if __name__ == "__main__":
    main() 