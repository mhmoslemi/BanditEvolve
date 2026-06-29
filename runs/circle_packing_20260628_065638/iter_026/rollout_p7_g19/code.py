import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Seed for deterministic initialization (for reproducibility)
    np.random.seed(42)
    
    # Initialize with a hexagonal grid with stochastic offsets to avoid symmetry
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.1, 0.1)  # increased stochasticity
        y = y_center + np.random.uniform(-0.1, 0.1)  # increased stochasticity
        # Alternate row staggering to optimize space
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.45 / cols - 7e-4  # slightly larger initial radii
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # same as before

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left side constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right side constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda for closure but with unique i and j
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Implement asymmetric reconfiguration: recompute a random subset of position offsets
    if res.success:
        v = res.x.copy()
        # Reintroduce random offsets to disrupt current grid and potentially unlock better positions
        for i in range(n):
            if np.random.rand() < 0.3:  # 30% of circles get repositioned
                row = i // cols
                col = i % cols
                x_center = (col + 0.5) / cols
                y_center = (row + 0.5) / rows
                offset = np.random.uniform(-0.15, 0.15)
                x = x_center + offset
                y = y_center + offset
                if row % 2 == 1:
                    x += 0.5 / cols
                v[3*i] = x
                v[3*i+1] = y
        
        # Re-optimize with new stochastic positions
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    # Targeted expansion of one circle
    if res.success:
        v = res.x.copy()
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])  # for distance computation
        
        # Calculate distances to all other circles per circle
        min_distances = np.zeros(n)
        for i in range(n):
            dists = np.sqrt((centers[:, 0] - centers[i, 0])**2 + (centers[:, 1] - centers[i, 1])**2)
            min_distances[i] = np.min(dists[dists > 1e-12])  # avoid zero distance
        
        # Identify the circle with the most "free space" (least constrained)
        least_constrained_idx = np.argmax(min_distances)
        
        # Apply a controlled expansion
        expand_ratio = 1.1
        # Expand the least constrained circle first
        new_radii = radii.copy()
        new_radii[least_constrained_idx] *= expand_ratio
        # Ensure not exceeding max allowed radius, and avoid overlap
        for j in range(n):
            if j != least_constrained_idx:
                dist = np.hypot(centers[least_constrained_idx, 0] - centers[j, 0],
                                centers[least_constrained_idx, 1] - centers[j, 1])
                if dist < new_radii[least_constrained_idx] + radii[j] - 1e-12:
                    new_radii[least_constrained_idx] = dist - radii[j] + 1e-12
        # Apply expansion to other circles proportionally
        for j in range(n):
            if j != least_constrained_idx:
                new_radii[j] *= (1 + (np.random.rand() * 0.02))  # small stochastic expansion
                if new_radii[j] > 0.5:
                    new_radii[j] = 0.5
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        # Re-evaluate with expanded radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    # Final check with perturbation to avoid local minima
    if res.success:
        v = res.x.copy()
        perturbation = np.random.rand(n, 2) * 0.03  # small random perturbation
        for i in range(n):
            v[3*i] += perturbation[i, 0]
            v[3*i+1] += perturbation[i, 1]
        
        # Validate that the perturbation hasn't moved the circles out of bounds
        for i in range(n):
            if v[3*i] < -1e-12 or v[3*i] + v[3*i+2] > 1 + 1e-12:
                v[3*i] = np.clip(v[3*i], 0.0, 1.0)
            if v[3*i+1] < -1e-12 or v[3*i+1] + v[3*i+2] > 1 + 1e-12:
                v[3*i+1] = np.clip(v[3*i+1], 0.0, 1.0)
        
        # Finally re-optimize
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # clip to max radius
    return centers, radii, float(radii.sum())