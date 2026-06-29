import numpy as np

def run_packing():
    n = 26
    
    # 1. Initialize with spatially aware hexagonal tiling and dynamic perturbation
    # We use 5 columns to create a 5x6 grid with hexagonal tiling
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize centers with hexagonal tiling and randomized perturbation
    xs = []
    ys = []
    
    # Generate a randomized "force field" for spatial activity
    spatial_activity = np.random.rand(n)
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        # For a hexagonal grid, we alternate between offset rows
        col_offset = (col_idx % 2 == 1) * 0.5 / cols
        x_center = (col_idx + 0.5 + col_offset) / cols
        # Use row index to space out rows more in a hex grid
        y_center = (row_idx + 0.5 + (row_idx % 2 * 0.5)) / rows
        # Add small, randomized perturbation to break symmetry
        x = x_center + np.random.uniform(-0.05, 0.05) * (1 + 1.5 * spatial_activity[i])
        y = y_center + np.random.uniform(-0.05, 0.05) * (1 + 1.5 * spatial_activity[i])
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with dynamic scaling based on spatial activity
    # We distribute more radius to highly active circles
    radii = 0.35 / cols * (1 + 0.5 * spatial_activity)
    radii = np.clip(radii, 1e-4, 0.35)  # clip to min and max reasonable radii
    
    v0 = np.zeros(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = radii
    
    # Ensure the bounds list has 3*n entries for the vector of length 3n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.45)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with spatial hashing and weight based on spatial activity
    
    # First create a spatial hashing map for efficient constraint handling
    spatial_hash = np.random.rand(n, 2) * 0.1
    # Create a weight matrix for constraint enforcement
    constraint_weights = np.zeros(n * n)
    for i in range(n):
        for j in range(i + 1, n):
            constraint_weights[i * n + j] = 1.0  # default weight
            constraint_weights[j * n + i] = constraint_weights[i * n + j]
    
    # Optimized constraint creation using NumPy for parallel processing
    def create_overlap_constraints():
        overlap_cons = []
        # Compute distance matrix using vectorization
        centers_mat = np.column_stack([v0[0::3], v0[1::3]])
        dx = centers_mat[:, np.newaxis, 0] - centers_mat[np.newaxis, :, 0]
        dy = centers_mat[:, np.newaxis, 1] - centers_mat[np.newaxis, :, 1]
        dist_squared = dx**2 + dy**2
        
        # Vectorize constraint creation
        for i in range(n):
            for j in range(i + 1, n):
                idx_i = i
                idx_j = j
                # Use spatial hashing to introduce variability in constraint strength
                hash_factor = 0.1 + 0.3 * np.random.rand()
                # Multiply by a weight based on spatial activity
                constraint_weight = constraint_weights[i * n + j] * (1.0 + 1.2 * spatial_activity[i])
                # Add the constraint with weighted return value
                overlap_cons.append({"type": "ineq", 
                                     "fun": lambda v, i=i, j=j: 
                                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                         - (v[3*i+2] + v[3*j+2])**2 * (1.0 + 0.01 * np.random.rand()),
                                     "jac": lambda v, i=i, j=j: (2*(v[3*i] - v[3*j]), 
                                                                 2*(v[3*i+1] - v[3*j+1]),
                                                                 -2*(v[3*i+2] + v[3*j+2])*0.999)})
        return overlap_cons
    
    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons + create_overlap_constraints(), 
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-12})
    
    # Step 2: Post-optimization refinement
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # 2.1 Compute critical circles and their spatial constraints
        # Vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dist_squared = dx**2 + dy**2
        
        # Compute minimum distance for each circle
        min_dist = np.min(dist_squared, axis=1)
        min_dist = np.sqrt(min_dist)
        
        # Identify the circles that contribute most to non-overlapping constraints
        critical_indices = np.argsort(min_dist)[:int(0.3 * n)]  # top 30% most constrained
        critical_idx_set = set(critical_indices)
        
        # 2.2 Enforce tight constraints on critical circles
        for i in critical_indices:
            # Force constraint: distance between circle i and any other circle
            for j in range(n):
                if i != j:
                    # Add a strict non-overlap constraint for this pair
                    # Using vectorization for efficiency
                    dx_ = v[3*i] - v[3*j]
                    dy_ = v[3*i+1] - v[3*j+1]
                    r_i = v[3*i+2]
                    r_j = v[3*j+2]
                    
                    # Add this as a new constraint
                    cons.append({"type": "ineq", "fun": lambda v, x=dx_, y=dy_, ri=r_i, rj=r_j:
                                 x*x + y*y - (ri + rj)**2})
        
        # Re-optimization with tight constraints
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-12})
    
    # Step 3: Targeted radius expansion based on spatial activity
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute minimum distance for each circle again
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dist_squared = dx**2 + dy**2
        min_dist = np.min(dist_squared, axis=1)
        min_dist = np.sqrt(min_dist)
        
        # Normalize spatial activity to [0, 1]
        normalized_activity = (spatial_activity - np.min(spatial_activity)) / (np.max(spatial_activity) - np.min(spatial_activity))
        
        # Calculate expansion factors based on:
        # - current radius
        # - minimal distance
        # - spatial activity
        
        # Initialize new radii with some base expansion
        new_radii = radii.copy()
        # Define expansion thresholds
        expansion_base = 0.004
        expansion_multiplier = 1.2
        min_dist_threshold = 0.05
        
        for i in range(n):
            # Calculate current expansion potential
            # Add more expansion to circles with low minimum distance
            # and higher spatial activity
            expansion_factor = expansion_base + expansion_multiplier * (min_dist[i] - min_dist_threshold)
            expansion_factor = np.clip(expansion_factor, 0.001, 0.01)  # cap to prevent overexpansion
            
            # Apply expansion, but only to circles that are not already maximal
            if radii[i] < 0.4:
                new_radii[i] += expansion_factor * (1.0 + 0.2 * normalized_activity[i])  # add spatial activity multiplier
                
        # Create a new decision vector with expanded radii
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Re-optimization with new radius configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-12})
    
    # Final clean-up and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())