import numpy as np

def run_packing():
    n = 26
    # Initialize with a novel geometric hashing mechanism: 
    # A dynamic lattice with probabilistic spatial constraints for better distribution
    cols = 5
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows, cols, 3))  # (x, y, radius) per grid cell
    for i in range(rows):
        for j in range(cols):
            grid[i][j][0] = (j + 0.5) / cols
            grid[i][j][1] = (i + 0.5) / rows
            grid[i][j][2] = 0.0
    # Apply randomized spatial hashing to generate a base grid with geometric constraints
    # Apply a novel "spatial perturbation map" that encodes geometric hashing for better convergence
    spatial_hash = np.random.rand(rows, cols, 3) * 0.15
    # Generate base positions with non-uniform distribution but avoiding clustering
    xs = []
    ys = []
    radii_initial = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols + spatial_hash[row][col][0] * 0.2
        base_y = (row + 0.5) / rows + spatial_hash[row][col][1] * 0.2
        # Apply geometric hashing of the base spatial location with a grid-aware weight
        grid_weight = (1.0 + (0.5 / (0.1 + (row + col))) ** 2)
        radius_base = 0.35 / cols * (1.0 / (1.0 + 0.2 * (row + col))) * grid_weight
        # Clamp to prevent initial over-expansion
        xs.append(base_x)
        ys.append(base_y)
        radii_initial.append(np.clip(radius_base, 1e-4, 0.5))
    
    # Generate initial decision vector with optimized geometric hashing and spatial constraints
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.array(radii_initial)

    # Construct a unified, clean and vectorized bounds list
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * n  # 3*n entries as needed

    def neg_sum_radii(v):
        """Objective function: minimize negative of total radius sum"""
        return -np.sum(v[2::3])
    
    # Generate constraints using lambda with i closures with fixed capturing
    # Boundary constraints
    cons = []
    for i in range(n):
        # Left boundary constraint: x - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: 1.0 - x - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: 1.0 - y - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with a geometric hashing optimization
    for i in range(n):
        for j in range(i + 1, n):
            # Create lambda constraint function with fixed captures for i,j
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # First optimization run: focused on geometric hashing with adaptive spatial perturbations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Apply geometric hashing-aware reconfiguration
    if res.success:
        v = res.x
        # Apply a grid-based spatial reconfiguration using a novel hashing matrix
        # Generate spatial "hash map" with adaptive scaling for better configuration space exploration
        spatial_map = np.random.normal(0, 0.02, n)  # For each circle's spatial perturbation
        # Apply targeted geometric hashing to positions
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += spatial_map[i] * (v[3*i] * (1.0 + 0.2 * np.random.rand()))
            new_v[3*i+1] += spatial_map[i] * (v[3*i+1] * (1.0 + 0.2 * np.random.rand()))
        
        # Second optimization with refined spatial hashing and reconfiguration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Implement a novel geometric hashing-aware topology adjustment
    if res.success:
        v = res.x
        # Compute pairwise distances with vectorized operations
        x_centers = v[0::3]
        y_centers = v[1::3]
        dx = x_centers[:, np.newaxis] - x_centers
        dy = y_centers[:, np.newaxis] - y_centers
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute geometric hashing matrix of minimum distances
        min_distances = np.min(dists, axis=1)
        # Find the most isolated circle in the geometric hashing matrix
        most_isolated_idx = np.argmax(min_distances)
        
        # Targeted expansion on the most isolated circle, with constraint-aware soft expansion
        # Compute growth factor based on current total and spatial constraints
        current_total = np.sum(v[2::3])
        avg_radius = current_total / n
        growth_factor = 0.003 * (avg_radius * 0.8)  # Soft expansion
        
        # Create enhanced radii array with targeted growth on the most isolated circle
        new_radii = v[2::3]
        if most_isolated_idx < n:  # Safety check
            new_radii[most_isolated_idx] += growth_factor * 1.5  # Slight over-expansion
            for i in range(n):
                if i != most_isolated_idx:
                    # Apply adaptive radius growth with geometric hashing
                    # Introduce geometric hashing-aware stochastic expansion
                    stochastic_expansion = (np.random.rand(n) * 0.3)
                    growth_i = growth_factor * (1.0 + 0.05 * stochastic_expansion[i])
                    new_radii[i] += growth_i
        # Apply clipping to prevent overflows
        new_radii = np.clip(new_radii, 1e-6, 0.5)
        
        # Create new decision vector for radius expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with radius expansion constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())