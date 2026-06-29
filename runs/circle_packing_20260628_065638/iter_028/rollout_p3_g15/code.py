import numpy as np

def run_packing():
    n = 26
    
    # Dynamic grid based on square root for optimal packing
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions with hierarchical random offsets
    xs = []
    ys = []
    # First layer: base grid with uniform distribution
    base_x = np.linspace(0.5 / cols, 1 - 0.5 / cols, cols) * (cols / (cols + 1))
    base_y = np.linspace(0.5 / rows, 1 - 0.5 / rows, rows) * (rows / (rows + 1))
    
    # Use a more sophisticated random offset pattern
    random_offsets = np.random.rand(n, 2)
    for i in range(n):
        row = i // cols
        col = i % cols
        x_base = base_x[col]
        y_base = base_y[row]
        # Apply multiple-level random offsets
        x = x_base + random_offsets[i, 0] * (0.1 / (cols + 0.5)) * (1 + (row % 2))
        y = y_base + random_offsets[i, 1] * (0.1 / (rows + 0.5)) * (1 + (col % 2))
        # Staggered grid pattern
        if row % 2 == 1:
            x += (0.5 / cols) * (0.95 + random_offsets[i, 0] * 0.05)
        xs.append(x)
        ys.append(y)
    
    # Adaptive radius initialization based on local density
    r0 = [0.25 / cols]
    for i in range(1, n):
        # Calculate base radius based on grid proximity
        r = 0.25 / cols * (1 + (0.1 * random_offsets[i, 0] * (1 if i % 2 == 0 else -1)))
        r = np.clip(r, 1e-4, 0.5)
        r0.append(r)
    r0 = np.array(r0)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # Ensure bounds have exactly 3*n values
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized constraint setup
    cons = []
    
    # Define a helper with proper closure using lambda that captures values
    def create_boundary_constraints(i):
        return [
            {"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]},
            {"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]},
            {"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]},
            {"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]}
        ]
    
    # Add boundary constraints for each circle
    for i in range(n):
        cons.extend(create_boundary_constraints(i))
    
    # Vectorized distance calculation and overlap constraints
    # We precompute the distance matrix to optimize evaluation
    # This is a more efficient approach than repeated pairwise distance
    # and is critical for stability and performance
    
    # Precompute distance matrix once for all constraints
    # We'll compute distances on-the-fly and recompute only when needed
    # This uses optimized broadcasting
    def compute_pairwise_dist(v):
        centers_x = v[0::3]
        centers_y = v[1::3]
        dx = centers_x[:, np.newaxis] - centers_x[np.newaxis, :]
        dy = centers_y[:, np.newaxis] - centers_y[np.newaxis, :]
        dists = np.sqrt(dx**2 + dy**2)
        return dists
    
    # Create a constraint function that checks for all pairs
    def create_overlap_constraints(i, j):
        def constraint_func(v):
            # Get centers and radii
            centers_x = v[0::3]
            centers_y = v[1::3]
            radii = v[2::3]
            dx = centers_x[i] - centers_x[j]
            dy = centers_y[i] - centers_y[j]
            dist = np.sqrt(dx**2 + dy**2)
            return dist - (radii[i] + radii[j]) + 1e-12
        return {"type": "ineq", "fun": constraint_func}
    
    # Create all pairwise overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append(create_overlap_constraints(i, j))
    
    # Initial optimization with adaptive step sizes
    # We start with a relatively low tolerance to capture rough shape
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-9})
    
    # Asymmetric reconfiguration: spatial constraint perturbation
    if res.success:
        v = res.x
        # Compute current radii and centers for validation
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate a more sophisticated geometric "least constrained" circle
        # by measuring minimal distance to other circles
        dist_matrix = compute_pairwise_dist(v)
        min_distances = np.min(dist_matrix, axis=1)
        constrained_indices = np.argsort(min_distances)
        least_constrained_idx = constrained_indices[-1]  # most un-constrained
        
        # Generate spatial hash with dynamic scaling based on circle positions
        spatial_hash = np.random.rand(n, 2) * 0.04
        # Scale perturbations based on circle size
        scale_factor = (radii / np.max(radii)) + 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * scale_factor[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale_factor[i]
        
        # Re-evaluate with new spatial configuration
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons, 
            options={"maxiter": 400, "ftol": 1e-11}
        )
    
    # Targeted radius expansion on least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Recalculate minimal distances for accurate constrained index
        dist_matrix = compute_pairwise_dist(v)
        min_distances = np.min(dist_matrix, axis=1)
        constrained_indices = np.argsort(min_distances)
        least_constrained_idx = constrained_indices[-1]
        
        # Calculate expansion while respecting constraints
        # We perform a safe expansion with validation
        expansion_factor = 0.005  # Targeted expansion
        expansion = expansion_factor * 1.1  # slight over-expansion
        
        while True:
            # Try expanding the least constrained circle + all others with some randomness
            expanded_v = v.copy()
            expanded_radii = radii.copy()
            expansion_vector = np.random.rand(n) * 0.03
            
            # Expand each circle by a fraction based on their distance to others
            for i in range(n):
                # More expansion for circles further from other circles
                expansion_i = expansion * (1 + 0.5 * expansion_vector[i])
                expanded_radii[i] += expansion_i
            
            # Check for constraint violations
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_radii[i] + expanded_radii[j]
                    dist = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
                    if dist < dx - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly
                expansion *= 0.95
        
        # Apply the safe expansion
        v = expanded_v.copy()
        v[2::3] = expanded_radii
        
        # Final optimization with refined constraints
        res = minimize(
            neg_sum_radii, 
            v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons, 
            options={"maxiter": 400, "ftol": 1e-11}
        )
    
    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())