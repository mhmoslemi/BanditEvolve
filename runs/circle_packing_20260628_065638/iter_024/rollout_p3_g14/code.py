import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized staggered grid and adaptive radius
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add stochastic offset to break symmetry
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Alternate row shift for staggered grid
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

    # Initial optimization with tighter tolerances and asymmetric configuration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Asymmetric reconfiguration: spatial constraint function with randomized gradient
    if res.success:
        v = res.x
        # Apply stochastic spatial perturbation to break symmetry
        random_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0] * 1.8
            perturbed_v[3*i+1] += random_hash[i, 1] * 1.8
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    # Targeted radius expansion on least constrained circle with controlled expansion
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
        # Calculate constraint isolation metric
        isolation = np.sum(1 / (dists + 1e-8), axis=1)
        isolated_idx = np.argmin(isolation)
        
        # Calculate expansion factor for controlled radius increase
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Adjust radii to gradually expand least constrained circle
        new_radii = radii.copy()
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] = min(radii[i] + expansion_factor * 1.1, 0.5)
        new_radii[isolated_idx] = min(radii[isolated_idx] + expansion_factor * 1.3, 0.5)
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and reconfiguration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Final refinement with controlled radius adjustments
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Calculate expansion potentials
        expansion_potentials = np.zeros(n)
        for i in range(n):
            # Count number of neighbors within 1.5*radius[i]
            neighbors = np.sum((dists[i] < 1.5 * radii[i]) & (dists[i] > 1e-8))
            expansion_potentials[i] = 1.0 / (neighbors + 1)  # less neighbors, higher potential
        
        # Expand the circle with highest expansion potential
        expansion_idx = np.argmax(expansion_potentials)
        
        # Calculate expansion factor for controlled radius increase
        expansion_factor = 0.0025 / (n - 1)
        new_radii = radii.copy()
        new_radii[expansion_idx] = min(radii[expansion_idx] + expansion_factor * 1.3, 0.5)
        for i in range(n):
            if i != expansion_idx:
                new_radii[i] = min(radii[i] + expansion_factor * 1.05, 0.5)
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final reevaluation
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())