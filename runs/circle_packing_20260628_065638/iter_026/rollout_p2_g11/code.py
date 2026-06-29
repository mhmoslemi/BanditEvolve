import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Use a more refined layout with dynamic spacing and random jitter
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add small random offset to break symmetry and reduce clustering
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Row staggering for staggered grid layout
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on spacing and small padding
    r0 = 0.29 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Set bounds consistent with 3*n parameters
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Boundary constraints: 4 per circle
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with fixed function definitions to avoid closure issues
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with high precision settings and increased max iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1600, "ftol": 1e-12, "eps": 1e-10})
    
    # Asymmetric spatial reconfiguration with controlled random perturbation
    if res.success:
        v = res.x
        spatial_hash = np.random.uniform(-0.07, 0.07, size=(n, 2))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-optimize with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-10})
    
    # Advanced targeting of isolated circle for expansion with soft constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute pairwise distances with broadcasting for isolation metric
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Find least constrained circle with isolation metric
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Compute target expansion and enforce soft constraints
        current_sum = np.sum(radii)
        target_sum = current_sum + 0.01
        expansion_factor = (target_sum - current_sum) / (n - 1)
        
        # Create expansion vector with stochasticity
        new_radii = radii.copy()
        expansion_weights = np.clip(1.0 + np.random.uniform(-0.3, 0.3, size=n), 0.5, 1.5)
       
        for i in range(n):
            if i == least_constrained_idx:
                new_radii[i] = np.clip(radii[i] + expansion_factor * expansion_weights[i], 1e-6, 0.5)
            else:
                new_radii[i] = np.clip(radii[i] + expansion_factor * expansion_weights[i] * 0.8, 1e-6, 0.5)

        # Apply expansion and re-optimize with constraints
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-10})
    
    # Final output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())