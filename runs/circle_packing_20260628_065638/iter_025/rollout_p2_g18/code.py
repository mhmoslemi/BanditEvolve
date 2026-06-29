import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with improved geometric clustering and dynamic offset
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset with adaptive radius-dependent scaling
        offset_scale = 0.15 * np.sqrt(1 / (rows * cols))
        x = x_center + np.random.uniform(-offset_scale, offset_scale)
        y = y_center + np.random.uniform(-offset_scale, offset_scale)
        # Staggered grid for even distribution
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.28 / cols - 1e-3
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

    # Vectorized overlap constraints with improved constraint handling
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12})

    # Stochastic reconfiguration with gradient-aware perturbation
    if res.success:
        v = res.x
        # Gradient-aware perturbation using directional randomization
        grad = np.gradient(v)
        directional_noise = np.random.rand(n, 2) * 0.05 * np.abs(grad[:n*2, :n*2].mean(axis=1, keepdims=True))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += directional_noise[i, 0]
            perturbed_v[3*i+1] += directional_noise[i, 1]
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Optimized radius expansion with gradient-based constraint adjustment
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation for all pairs
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by minimal minimum distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Gradient-based expansion with radius-aware constraint reordering
        # Calculate expansion factor using weighted average
        expansion_factor = 0.0075 / (n - 1)  # Base factor
        weight = 1.0 + 0.5 * np.random.rand()  # Introduce controlled randomness
        expansion_factor *= weight
        
        # Create optimized expansion vector
        new_radii = radii.copy()
        # Apply expansion to least constrained and minimal radius circle more significantly
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        new_radii[np.argmin(radii)] += expansion_factor * 1.3
        for i in range(n):
            if i not in [least_constrained_idx, np.argmin(radii)]:
                new_radii[i] += expansion_factor * 0.9
        
        # Apply expansion to a new decision vector with constraint validation
        v_new = v.copy()
        v_new[2::3] = new_radii

        # Use warm start with tighter tolerance and gradient-based adjustment
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12,
                                                  "jac": np.gradient(v_new) * 1e-5})

    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())