import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric tiling and spatial clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid points
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        # Add randomized offsets for spatial hashing
        x = x_base + np.random.uniform(-0.06, 0.06)
        y = y_base + np.random.uniform(-0.06, 0.06)
        # Stagger rows for hexagonal grid approximation
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.4 / cols - 1e-2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints
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
    
    # Topological shift: apply randomized geometric tiling to disrupt configuration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric tiling pattern
        tiling_hash = np.random.rand(n, 2) * 0.05
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += tiling_hash[i, 0] * (1.0 - v[3*i+2])
            new_v[3*i+1] += tiling_hash[i, 1] * (1.0 - v[3*i+2])
        
        # Re-evaluate with new geometric tiling
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    # Targeted radius expansion on the least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation (optimized for efficiency)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle
        min_dist = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dist)
        
        # Calculate controlled expansion factor based on proximity to edge
        edge_proximity = np.min([v[3*least_constrained_idx] + 1e-3,
                                 1.0 - v[3*least_constrained_idx] - v[3*least_constrained_idx+2],
                                 1.0 - v[3*least_constrained_idx+1] - v[3*least_constrained_idx+2],
                                 v[3*least_constrained_idx] - v[3*least_constrained_idx+2]])
        expansion_factor = (0.008 / (n - 1)) * (1.0 - edge_proximity) / 1.0
        
        # Expand the least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.4
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector and re-optimize
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())