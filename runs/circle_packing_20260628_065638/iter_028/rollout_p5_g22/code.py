import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Step 1: Initialize positions using a staggered grid with spatial randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add random spatial jitter to break symmetry
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Stagger rows for better packing
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimate based on grid density and perturbation
    mean_radius = 0.35 / np.sqrt(n) - 1e-3
    r0 = mean_radius + np.random.uniform(-0.01, 0.01) * mean_radius
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for position and radii
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * n

    # Objective function to maximize sum of radii (minimize negative sum)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint functions for boundaries
    def boundary_constraints(v, i):
        x = v[3*i] - v[3*i+2]  # Left + radius <= 1
        y = 1.0 - v[3*i] - v[3*i+2]  # Right - radius >= 0
        top = 1.0 - v[3*i+1] - v[3*i+2]
        bottom = v[3*i+1] - v[3*i+2]
        return np.array([x, y, top, bottom])

    # Vectorized overlap constraints using square of Euclidean distances
    def overlap_constraints(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        dist = np.hypot(dx, dy)
        return dist - (v[3*i+2] + v[3*j+2])

    # Initial optimization with global exploration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": lambda v, i=i: boundary_constraints(v, i)[0]}
                               for i in range(n)] +
                              [{"type": "ineq", "fun": lambda v, i=i: boundary_constraints(v, i)[1]}
                               for i in range(n)] +
                              [{"type": "ineq", "fun": lambda v, i=i: boundary_constraints(v, i)[2]}
                               for i in range(n)] +
                              [{"type": "ineq", "fun": lambda v, i=i: boundary_constraints(v, i)[3]}
                               for i in range(n)] +
                              [{"type": "ineq", "fun": lambda v, i=i, j=j: overlap_constraints(v, i, j)}
                               for i in range(n) for j in range(i + 1, n)],
                   options={"maxiter": 1500, "ftol": 1e-10, "disp": False})
    
    v = res.x if res.success else v0

    # Step 2: Spatial reconfiguration with radius-based perturbation
    if res.success:
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Adaptive radius-based spatial perturbation
        radius_scaling = np.mean(radii) * 0.08  # Scale perturbation by radius size
        spatial_hash = np.random.randn(n, 2) * radius_scaling
        v_perturbed = v.copy()
        v_perturbed[0::3] += spatial_hash[:, 0]
        v_perturbed[1::3] += spatial_hash[:, 1]
        
        # Re-evaluate with perturbed positions
        res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 400, "ftol": 1e-10})
        
        v = res.x if res.success else v

    # Step 3: Identify least constrained circle
    if res.success:
        # Vectorized distance computation using broadcasting
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Compute minimum distance to other circles
        min_dists = np.min(distances, axis=1)
        isolated_idx = np.argmax(min_dists)
        base_radius = radii[isolated_idx]
        avg_radius = np.mean(radii)

        # Targeted radius expansion with dynamic coefficient
        expansion_coefficient = 0.3 * (avg_radius / base_radius) * (np.max(min_dists) / (np.min(min_dists) + 1e-8))
        expansion = expansion_coefficient * 0.3
        expansion = np.clip(expansion, 0.001, 0.01)  # Limit controlled expansion
        
        # Create new radius array with expansion on least constrained circle
        new_radii = radii.copy()
        new_radii[isolated_idx] += expansion
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion * (1.0 + np.random.uniform(-0.1, 0.1)) * (1.0 if radii[i] > base_radius else 0.8)
        
        # Apply new radii and re-optimize
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 400, "ftol": 1e-10})
        v = res.x if res.success else v

    # Final refinement and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())