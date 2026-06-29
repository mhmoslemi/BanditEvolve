import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with a base staggered grid and geometric hashing for asymmetry
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on grid spacing
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bounds, consistent with 3n variables

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left and right boundary constraints (x - r >= 0 and x + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom and top boundary constraints (y - r >= 0 and y + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with geometric hashing
    geometric_hash = np.random.rand(n, 2) * 0.08  # Introduce controlled randomness for layout disruption

    # First optimization with high tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "gtol": 1e-9})

    if res.success:
        v = res.x
        # Apply asymmetric displacement using geometric hash for topological reconfiguration
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += geometric_hash[i, 0]
            perturbed_v[3*i+1] += geometric_hash[i, 1]
        
        # Additional constraints for stronger topology disruption (minimal spacing enforced)
        for i in range(n):
            for j in range(i + 1, n):
                # Enforce minimal spacing between adjacent circles with controlled margin
                cons.append({"type": "ineq", 
                             "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + 
                                                       (v[3*i+1] - v[3*j+1])**2 - 
                                                       (v[3*i+2] + v[3*j+2])**2 + 1e-5})
        
        # Re-evaluate with perturbed parameters and enhanced constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-10})

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
        
        # Find the least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor based on geometric hashing and layout stability
        total_sum = np.sum(radii)
        expansion_factor = (0.012 / (n - 1)) * (1 + 0.1 * np.random.rand())  # Stochastic controlled expansion
        
        # Expand radii asymmetrically while maintaining constraints
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.5  # Over-expand least constrained
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())