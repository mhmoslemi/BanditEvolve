import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with staggered grid, adaptive jitter, and adaptive radius scaling
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base positions: staggered grid with row-based offset
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Jitter with adaptive amplitude based on radius potential
        jitter_range = 0.035 * (0.5 / cols)  # Adjusted for better initial diversity
        x = x_center + np.random.uniform(-jitter_range, jitter_range)
        y = y_center + np.random.uniform(-jitter_range, jitter_range)
        
        # Stagger alternate rows to prevent uniform clustering
        if row % 2 == 1:
            x += 0.4 / cols  # Adjusted stagger amplitude for better spacing
        xs.append(x)
        ys.append(y)
    
    # Set initial radii with higher base and better radius distribution
    base_radius = 0.35 / cols * 1.1  # Increased base radius for better growth potential
    r0 = np.full(n, base_radius) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Strict bounds that match the decision vector length exactly
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3n ensures consistency

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Generate constraints using lambda with captured i to ensure proper indexing
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})

    # Overlap constraints with efficient vectorization to reduce computation cost
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-9})  # Sufficient tolerance
    
    # Perturbation step: shake low-impact circles to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        
        # Identifying circles with minimal influence for shaking
        impact_mask = np.zeros(n, dtype=bool)
        # Circle is considered "shakable" if its radius is small and it has ample space around
        # This is a heuristic based on distance to neighbors and bounding constraints
        for i in range(n):
            # Check proximity to edges
            if v[3*i] - v[3*i+2] < 0.05 or 1.0 - v[3*i] - v[3*i+2] < 0.05:
                impact_mask[i] = True
            if v[3*i+1] - v[3*i+2] < 0.05 or 1.0 - v[3*i+1] - v[3*i+2] < 0.05:
                impact_mask[i] = True
            
            # Check proximity to other circles
            dx, dy = np.abs(v[3*i] - v[3::3]), np.abs(v[3*i+1] - v[4::3])
            min_dist = np.minimum(dx, dy).min()
            if min_dist < 0.1:
                impact_mask[i] = True
            elif np.random.rand() < 0.3:
                impact_mask[i] = True
        
        # Apply shaking only to low-impact circles
        for i in np.where(impact_mask)[0]:
            # Perturb center with small random offset
            max_move = 0.01 * np.sqrt(v[3*i+2])  # Movement scales with radius
            v[3*i] += np.random.uniform(-max_move, max_move)
            v[3*i+1] += np.random.uniform(-max_move, max_move)
        
        # Re-run optimization with perturbed configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10})

    # Advanced targeted expansion with spatial prioritization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distances to neighbors and edges
        min_edge_dists = np.zeros(n)
        for i in range(n):
            min_edge_dists[i] = min(
                v[3*i] - radii[i],
                1.0 - v[3*i] - radii[i],
                v[3*i+1] - radii[i],
                1.0 - v[3*i+1] - radii[i]
            )
        
        # Spatial fitness: higher fitness means more expansion potential
        spatial_fitness = min_edge_dists + np.sqrt(np.sum(centers**2, axis=1)) * 0.3
        
        # Compute circle-to-circle distances and determine expansion potential
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                distance_matrix[i, j] = np.sqrt(dx**2 + dy**2)
        
        # Identify most expansion-friendly circles: those with large distance to neighbors and edges
        expansion_indices = np.argsort(spatial_fitness + (1.0 - distance_matrix.min(axis=1)) * 10)[::-1]
        
        # Targeted expansion
        expansion_amount = 0.005  # Base expansion
        expansion_factor = np.zeros(n)
        
        for i in expansion_indices:
            # Calculate maximum expansion based on available space
            max_radius = 0
            if v[3*i] - radii[i] < 0.01:
                max_radius = 0  # Too close to left edge
            elif 1.0 - v[3*i] - radii[i] < 0.01:
                max_radius = 0  # Too close to right edge
            if v[3*i+1] - radii[i] < 0.01:
                max_radius = 0  # Too close to bottom edge
            elif 1.0 - v[3*i+1] - radii[i] < 0.01:
                max_radius = 0  # Too close to top edge
            
            # If not too close to edges, expand based on available space
            if max_radius > 0:
                # Expand only if circle is not too small
                if radii[i] < 0.01 * 0.5:
                    # Limit expansion for very small circles
                    expansion_amount = 0
                else:
                    expansion_amount = max(0.0001, min(0.01, (max_radius - 0.01) * 3))
                expansion_factor[i] = expansion_amount
        
        # Apply expansion, keeping radius bounds
        for i in range(n):
            if expansion_factor[i] > 0:
                v[3*i + 2] = min(v[3*i + 2] + expansion_factor[i], 0.45)  # Max radius bound
            
        # Re-optimize the expanded configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())