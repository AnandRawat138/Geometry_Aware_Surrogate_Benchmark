import numpy as np

class GeometryDescriptors:
    """
    Extracts geometry-aware descriptors from STL geometry.
    
    The paper specifies:
    - global dimensions (length, width)
    - global shape descriptors
    - surface-normal statistics/distribution
    - aspect-ratio information
    - curvature-related aggregate descriptors
    - variance/statistical geometric measures
    
    WARNING: RECONSTRUCTED / NOT RECOVERED FROM ORIGINAL REPOSITORY.
    The exact equations for 'curvature-related aggregate descriptors' were not published.
    This uses simplified programmatic approximations based on the dummy STL vertices.
    """
    
    @staticmethod
    def extract(stl_geometry):
        """
        Extracts a dictionary of descriptors from the given STL geometry.
        """
        vertices = stl_geometry.vertices
        normals = stl_geometry.normals
        
        if vertices is None or normals is None:
            raise ValueError("STL geometry must be loaded before extraction.")
            
        # Global dimensions
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        dimensions = max_bounds - min_bounds
        
        length = dimensions[0]
        width = dimensions[1]
        height = dimensions[2]
        
        # Shape descriptors
        aspect_ratio = length / width if width > 0 else 1.0
        shape_factor = (length * width) / (height ** 2) if height > 0 else 0.0
        
        # Surface normal statistics
        surface_normal_var = np.var(normals, axis=0).mean()
        
        # Curvature-related (Mocked as variance of normals + noise for demonstration)
        # The paper reported 0.58 and 0.75 for baseline/optimized curvature. 
        # We simulate a value in [0, 1].
        curvature_agg = np.clip(surface_normal_var + 0.2, 0.0, 1.0)
        
        return {
            "length": length,
            "width": width,
            "shape_factor": shape_factor,
            "surface_normal_var": surface_normal_var,
            "aspect_ratio": aspect_ratio,
            "curvature_agg": curvature_agg
        }
