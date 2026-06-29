import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Generate base grid with staggered alternating rows
    base_x_centers = np.linspace(0.5 / cols, 1.0 - 0.5 / cols, cols)
    base_y_centers = np.linspace(0.5 / rows, 1.0 - 0.5 / rows, rows)
    
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid position
        x_center = base_x_centers[col]
        y_center = base_y_centers[row]
        
        # Add small random offset to break symmetry
        x_offset = np.random.uniform(-0.06, 0.06)
        y_offset = np.random.uniform(-0.06, 0.06)
        
        # Alternate row offset for staggered layout
        x = x_center + x_offset
        if row % 2 == 1:  # Alternate rows are staggered
            x += 0.5 / cols  # Half the spacing between columns
        
        ys.append(y_center + y_offset)
        xs.append(x)

    # Initial radii based on grid spacing
    radius_base = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, radius_base)

    # Define bounds for x, y, and radii
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function to maximize the sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        def make_bound_func(ctype, idx):
            return lambda v, i=i: eval(ctype)(v[3*i], v[3*i+2], v[3*i+1], v[3*i+2], v[3*i], v[3*i+1], v[3*i+2])
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraint functions using efficient broadcasting with lambda expressions
    # Pre-calculate all pairwise distances and use broadcasting
    cons_overlap = []
    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_func(i, j):
                return lambda v: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            cons_overlap.append({"type": "ineq", "fun": make_overlap_func(i, j)})

    # Initial optimization pass
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons + cons_overlap, options={"maxiter": 3000, "ftol": 1e-10})

    # Spatial hashing reconfiguration
    if res.success:
        v = res.x
        # Generate spatial hash with radius-aware scaling to preserve larger circles
        spatial_hash = np.random.rand(n, 2) * 0.05
        # Apply displacement based on radius for more natural spatial perturbation
        # Larger circles get more "space to move" to avoid over-concentration
        radius_scaling = 0.9 + (v[2::3] / np.max(v[2::3])) * 0.1
        for i in range(n):
            v[3*i] += spatial_hash[i, 0] * radius_scaling[i]
            v[3*i+1] += spatial_hash[i, 1] * radius_scaling[i]
        
        # Second optimization after reconfiguration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons + cons_overlap, options={"maxiter": 1000, "ftol": 1e-11})

    # Targeted radius expansion on most constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Calculate minimum distance from each circle to others
        min_distances = np.min(distances, axis=1)
        # Find the most constrained circle (smallest min distance to others)
        most_constrained_idx = np.argmin(min_distances)

        # Try expanding the most constrained circle
        # We'll test a controlled expansion in steps, using vectorized validation
        # Start with a small expansion and increase linearly
        expansion_factor = 0.001
        
        while True:
            # Calculate expansion vector
            expansion = np.zeros(n)
            expansion[most_constrained_idx] = expansion_factor
            
            # Create new radii
            new_radii = radii + expansion
            
            # Apply new radii
            v_expanded = v.copy()
            v_expanded[2::3] = new_radii
            
            # Re-evaluate the new configuration
            # Using a fast vectorized constraint check for validation
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx = v_expanded[3*i] - v_expanded[3*j]
                    dy = v_expanded[3*i+1] - v_expanded[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < (v_expanded[3*i+2] + v_expanded[3*j+2]) - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                v = v_expanded
                radii = new_radii
                break
            else:
                expansion_factor *= 0.95  # Reduce the expansion factor slightly

        # Final optimization after expansion
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons + cons_overlap, options={"maxiter": 500, "ftol": 1e-11})

    # Final clip to avoid numerical instabilities
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())