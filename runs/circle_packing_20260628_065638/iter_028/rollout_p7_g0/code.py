import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Advanced initializer: hybrid of grid, symmetry-breaking, and dynamic spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Use a non-uniform offset to break symmetry with spatial bias
        x_offset = np.random.normal(0, 0.03 * (1 + row * 0.2))
        y_offset = np.random.normal(0, 0.03 * (1 + row * 0.2)) 
        x = base_x + x_offset
        y = base_y + y_offset
        
        # Stagger alternate rows with spatial gradient
        if row % 2 == 1:
            x += 0.5 / cols * (row / (rows - 1))  # Gradually increase row spacing
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with delayed lambda binding
    cons = []
    for i in range(n):
        # Bound constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bound constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bound constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Bound constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Smart overlap constraints with adaptive scaling
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r_sum = v[3*i+2] + v[3*j+2]
                # Use adaptive scaling to help optimization
                return dist_sq - (r_sum * (1 + 0.05 * np.random.rand()))
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tight precision and hybrid method
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11,
                                              "gtol": 1e-11, "eps": 1e-11})
    
    # Geometric dissection on most interacting circles: identified through adaptive analysis
    if res.success:
        v = res.x
        radii = v[2::3]
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting for speed
        dx = v[0::3] - v[0::3][:, np.newaxis]
        dy = v[1::3] - v[1::3][:, np.newaxis]
        dists = np.sqrt(dx**2 + dy**2)
        interaction_scores = np.sum(dists, axis=1)
        
        # Identify most dynamically interacting circles and least constrained
        top_interactors = np.argsort(interaction_scores)[-3:]
        isolation_scores = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(isolation_scores)
        
        # Create custom perturbation that isolates top interactors and expands least constrained
        # Use controlled geometric dissection to restructure their spatial relationships
        perturbed_v = v.copy()
        for i in top_interactors:
            # Random displacement with spatial scaling to avoid symmetry
            displacement_factor = 0.02 * (1 + np.random.rand()) * v[3*i]  # Spatially scaled
            perturbed_v[3*i] += np.random.uniform(-displacement_factor, displacement_factor)
            perturbed_v[3*i+1] += np.random.uniform(-displacement_factor, displacement_factor)
        # Expand least constrained circle gradually with spatial scaling
        expansion_factor = 0.01 * (1 + np.random.rand()) * v[3*least_constrained_idx]
        perturbed_v[3*least_constrained_idx + 2] += np.random.uniform(expansion_factor*0.5, expansion_factor*1.5)
        
        # Re-evaluate with this reconfiguration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11,
                                                  "gtol": 1e-11, "eps": 1e-11})
    
    # Multi-stage spatial expansion on least constrained with adaptive enforcement
    if res.success:
        v = res.x
        radii = v[2::3]
        dists = np.zeros((n, n))
        dx = v[0::3] - v[0::3][:, np.newaxis]
        dy = v[1::3] - v[1::3][:, np.newaxis]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Recalculate isolation score and choose least constrained
        isolation_scores = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(isolation_scores)
        
        # Compute potential expansion based on total current sum and distance margins
        current_total = np.sum(radii)
        expansion_ratio = 0.0065 / (n - 1) * (current_total / np.sum(radii))
        expansion_factor = expansion_ratio * (1 + np.random.rand())  # Stochastic expansion
        
        # Attempt expansion with spatial constraint adaptation
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1 + np.random.rand() * 0.3)
                new_radii[i] += expansion_i
        
        # Constraint-based validation and fallback mechanism
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # Cooldown strategy for invalid expansion
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11,
                                                  "gtol": 1e-11, "eps": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())