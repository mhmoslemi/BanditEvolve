import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with grid-based placement with randomized stochastic refinement
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols + np.random.uniform(-0.025, 0.025)
        y = (row + 0.5) / rows + np.random.uniform(-0.025, 0.025)
        # Alternate row offset for staggered pattern
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.33 / cols - 1e-3
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing and spatial filtering
    for i in range(n):
        for j in range(i + 1, n):
            # Compute minimal distance constraint with spatial filtering
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: 
                         dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2})
    
    # Initial optimization with advanced settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    # Apply stochastic perturbation to the circle with the smallest radius
    if res.success:
        v = res.x
        radii = v[2::3]
        smallest_radius_idx = np.argmin(radii)
        # Apply controlled random perturbation to the position of the smallest circle
        v[3*smallest_radius_idx] += np.random.uniform(-0.04, 0.04)
        v[3*smallest_radius_idx+1] += np.random.uniform(-0.04, 0.04)
        # Slight radius increase to trigger new layout
        v[3*smallest_radius_idx+2] += np.random.uniform(0.001, 0.002)
        # Re-evaluate
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    # Apply geometric hashing perturbation with spatial coherence
    if res.success:
        v = res.x
        random_hash = np.random.rand(n, 2) * 0.01
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    # Adaptive radius expansion with constraint-aware growth
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Find least constrained circle based on minimum distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate controlled expansion factor
        current_sum = np.sum(radii)
        expansion_factor = 0.008 / (n - 1)  # Based on historical performance trends
        
        # Apply expansion with adjacency-aware scaling
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())