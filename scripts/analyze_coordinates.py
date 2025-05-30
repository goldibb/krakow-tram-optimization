import geopandas as gpd
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_file(file_path):
    logger.info(f"Starting analysis of {file_path}...")
    try:
        logger.info("Reading file...")
        gdf = gpd.read_file(file_path)
        logger.info(f"File read successfully. Number of features: {len(gdf)}")
        
        logger.info("Converting to EPSG:2180...")
        gdf_projected = gdf.to_crs(epsg=2180)
        logger.info("Conversion completed")
        
        # Get bounds of all geometries
        bounds = gdf_projected.total_bounds
        
        logger.info(f"X range: {bounds[0]:.2f} to {bounds[2]:.2f}")
        logger.info(f"Y range: {bounds[1]:.2f} to {bounds[3]:.2f}")
        
        return bounds
    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {str(e)}")
        raise

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    logger.info(f"Data directory: {data_dir}")
    
    files = ['buildings.geojson', 'streets.geojson', 'stops.geojson', 'lines.geojson']
    
    all_bounds = []
    for file in tqdm(files, desc="Processing files"):
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            continue
            
        logger.info(f"\nProcessing {file}...")
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