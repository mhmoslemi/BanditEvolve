import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with a randomized staggered grid and asymmetric offset
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Asymmetric random offset to break symmetry
        x = x_center + np.random.uniform(-0.03, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.03)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.35 / cols
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
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Asymmetric reconfiguration: apply stochastic spatial perturbation to all circles
    if res.success:
        v = res.x
        # Apply symmetric random offset to all circles to induce structural change
        perturbation = np.random.uniform(-0.015, 0.015, size=(n, 2))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation[i, 0]
            perturbed_v[3*i+1] += perturbation[i, 1]
        
        # Re-evaluate with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion on least constrained circle based on minimum distance
    if res.success:
        v = res.x
        radii = v[2::3]
        # Compute distance matrix for all pairwise circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Find least constrained circle (largest minimum distance to others)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        # Compute current total sum
        total_sum = np.sum(radii)
        # Calculate expansion factor for controlled radius increase
        target_total_sum = total_sum + 0.009
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Create adjusted radius vector with asymmetric expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Boost expansion factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    # Final refinement pass with tighter tolerances
    if res.success:
        v = res.x
        radii = v[2::3]
        # Compute distance matrix for all pairwise circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Recalculate constraint tightness (distance to edges and other circles)
        constraint_tightness = np.zeros(n)
        for i in range(n):
            # Distance to edges
            constraint_tightness[i] += (1.0 - v[3*i] - v[3*i+2]) + (1.0 - v[3*i+1] - v[3*i+2])
            # Distance to other circles
            for j in range(n):
                if i != j:
                    constraint_tightness[i] += max(0, radii[i] + radii[j] - dists[i, j])
        # Identify the top 3 most constrained circles for reconfiguration
        most_constrained = np.argsort(constraint_tightness)[-3:]
        # Apply small random perturbations to their positions
        for idx in most_constrained:
            perturbation = np.random.uniform(-0.005, 0.005, size=2)
            v[3*idx] += perturbation[0]
            v[3*idx+1] += perturbation[1]
        
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())