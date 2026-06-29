import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply shake heuristic to smallest circles to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify the smallest circles to shake
        smallest_indices = np.argsort(radii)[:5]
        # Apply small random perturbations to their positions
        for i in smallest_indices:
            v[3*i] += np.random.uniform(-0.04, 0.04)
            v[3*i+1] += np.random.uniform(-0.04, 0.04)
            v[3*i+2] += np.random.uniform(-0.003, 0.003)
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Radical topological reconfiguration using geometric hashing with adjacency constraints
    if res.success:
        v = res.x
        # Generate a new, randomized geometric configuration that enforces strict non-overlap
        spatial_hash = np.random.rand(n, 2) * 0.04
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += spatial_hash[i, 0]
            new_v[3*i+1] += spatial_hash[i, 1]
        
        # Enforce strict boundary and non-overlap constraints
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # Targeted radius expansion on the circle with the smallest non-zero radius
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances and identify the circle with the smallest non-zero radius
        min_distance = np.inf
        smallest_radius_idx = -1
        for i in range(n):
            if radii[i] < 1e-6:
                continue
            dx = centers[i, 0] - centers[np.arange(n)]
            dy = centers[i, 1] - centers[np.arange(n), 1]
            dists = np.sqrt(dx*dx + dy*dy)
            nearby_distances = np.sort(dists)
            min_distance = min(min_distance, nearby_distances[1])
            if radii[i] < 1e-6:
                continue
            if np.min(dists) > min_distance:
                smallest_radius_idx = i
        
        # If we found a non-zero radius circle, perform targeted expansion
        if smallest_radius_idx != -1:
            # Calculate expansion factor to gradually increase radius with strict non-overlap
            total_sum = np.sum(radii)
            expansion_factor = 0.01 / (n - 1)  # Controlled expansion to unlock new configuration
            
            # Create adjusted radius vector with adjacency-based expansion
            new_radii = radii.copy()
            new_radii[smallest_radius_idx] += expansion_factor * 1.5  # Slight over-expansion
            for i in range(n):
                if i != smallest_radius_idx:
                    new_radii[i] += expansion_factor
            
            # Update decision vector and re-evaluate with new constraints
            v_new = v.copy()
            v_new[2::3] = new_radii
            
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())