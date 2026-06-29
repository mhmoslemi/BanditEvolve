import numpy as np

def run_packing():
    n = 26
    cols = 5
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
    
    # Apply targeted 'shake' heuristic with adaptive perturbation and constraint-aware expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify the smallest circles and a few randomly selected ones for perturbation
        small_radius_indices = np.argsort(radii)[:5]
        random_indices = np.random.choice(n, size=5, replace=False)
        perturbation_indices = np.union1d(small_radius_indices, random_indices)
        
        # Apply small random perturbations with adaptive magnitude and constraint-aware adjustment
        for idx in perturbation_indices:
            # Compute min distance to edges and other circles
            min_edge_dist = min(v[3*idx] - v[3*idx+2], 1.0 - v[3*idx] - v[3*idx+2],
                               v[3*idx+1] - v[3*idx+2], 1.0 - v[3*idx+1] - v[3*idx+2])
            min_circle_dist = np.inf
            for j in range(n):
                if j != idx:
                    dx = v[3*idx] - v[3*j]
                    dy = v[3*idx+1] - v[3*j+1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < min_circle_dist:
                        min_circle_dist = dist
            # Calculate adaptive perturbation magnitude based on constraint tightness
            perturbation_magnitude = np.max([min_edge_dist, min_circle_dist]) * 0.1
            perturbation = np.random.uniform(-perturbation_magnitude, perturbation_magnitude, size=2)
            v[3*idx] += perturbation[0]
            v[3*idx+1] += perturbation[1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Radical geometric reconfiguration through constraint-aware spatial hashing
    if res.success:
        v = res.x
        # Create a constraint-aware geometric hash map for new configuration
        random_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            # Apply geometric hashing with constraint awareness
            if v[3*i+2] > 1e-6:
                perturbation = np.random.uniform(-0.02, 0.02, size=2) * (1 + np.random.rand())
                perturbed_v[3*i] += perturbation[0]
                perturbed_v[3*i+1] += perturbation[1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion on smallest circle with adjacency-based expansion and constraint-aware adjustment
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the circle with the smallest non-zero radius
        smallest_radius_idx = np.argmin(radii)
        # Check if the radius is non-zero
        if radii[smallest_radius_idx] < 1e-6:
            smallest_radius_idx = np.argmin(np.abs(radii - 1e-6))
        
        # Calculate distances from other circles and edges for constraint awareness
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Calculate constraint tightness for the smallest circle
        constraint_tightness = 0
        for j in range(n):
            if j != smallest_radius_idx:
                constraint_tightness += max(0, radii[smallest_radius_idx] + radii[j] - dists[smallest_radius_idx, j])
        constraint_tightness += (1.0 - v[3*smallest_radius_idx] - radii[smallest_radius_idx])
        constraint_tightness += (1.0 - v[3*smallest_radius_idx+1] - radii[smallest_radius_idx])
        
        # Calculate expansion factor for controlled radius increase
        expansion_factor = 0.005 / (1 + constraint_tightness * 0.1)
        
        # Create adjusted radius vector with adjacency-based expansion
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] += expansion_factor * 1.2  # Slight over-expansion to trigger reconfiguration
        for i in range(n):
            if i != smallest_radius_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new adjacency constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())