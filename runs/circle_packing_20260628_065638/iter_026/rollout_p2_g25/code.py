import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with staggered grid and adaptive offset
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Adaptive offset to introduce asymmetry
        x = x_center + np.random.uniform(-0.05, 0.05) * (1.0 + 0.2 * np.sin(row * 0.5))
        y = y_center + np.random.uniform(-0.05, 0.05) * (1.0 + 0.2 * np.cos(row * 0.5))
        # Row staggering for better spacing
        if row % 2 == 1:
            x += 0.4 / cols * np.random.uniform(-1.0, 1.0)
        xs.append(x)
        ys.append(y)
    
    # Initial radius based on spacing and some padding with dynamic scaling
    r0 = (0.35 / cols) + (0.4 / cols) * (1.0 - 0.5 * np.sin(row * 0.5)) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define consistent bounds for all 3 * n parameters
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Define boundary constraints with strict closure handling
    cons = []
    for i in range(n):
        # Left + radius <= 1
        def fun_left(v, i=i):
            return 1.0 - v[3*i] - v[3*i+2]
        cons.append({"type": "ineq", "fun": fun_left})
        # Right - radius >= 0
        def fun_right(v, i=i):
            return v[3*i] - v[3*i+2]
        cons.append({"type": "ineq", "fun": fun_right})
        # Bottom + radius <= 1
        def fun_bottom(v, i=i):
            return 1.0 - v[3*i+1] - v[3*i+2]
        cons.append({"type": "ineq", "fun": fun_bottom})
        # Top - radius >= 0
        def fun_top(v, i=i):
            return v[3*i+1] - v[3*i+2]
        cons.append({"type": "ineq", "fun": fun_top})

    # Define vectorized distance-based overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Avoid closure issues by using lambda with parameter capture
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization: base pattern
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-12})

    # Asymmetric reconfiguration step: trigger localized spatial perturbation
    if res.success:
        v = res.x
        # Create a more refined spatial hash for asymmetric reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (1.0 + 0.2 * np.sin(i * 0.5))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (1.0 + 0.2 * np.cos(i * 0.5))
        
        # Re-evaluate with perturbed spatial coordinates
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Refine further by expanding the least constrained circle with dynamic expansion factor
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Use broadcasting for fast distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the least constrained circle (maximum isolation)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion amount with dynamic, soft enforcement
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.007  # Incremented from 0.006 for more room
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        expansion_decay = 0.95  # Controlled expansion decay
        expansion_factor *= (1.0 + 0.2 * np.random.rand())  # Stochastic variation
        
        # Apply expansion to the least constrained circle first
        new_radii = radii.copy()
        new_radii[least_constrained_idx] = np.clip(
            radii[least_constrained_idx] + expansion_factor * 1.2, 
            1e-4, 0.5
        )
        
        # Apply moderate expansion to others with decay
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] = np.clip(
                    radii[i] + expansion_factor * 0.8 * expansion_decay,
                    1e-4, 0.5
                )
        
        # Apply expansion and validate
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Use local reevaluation for efficiency
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-12})
    
    # Final configuration after all refinements
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())