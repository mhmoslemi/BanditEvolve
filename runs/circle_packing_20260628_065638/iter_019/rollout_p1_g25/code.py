import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
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

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Radical reconfiguration with geometric hashing
    if res.success:
        v = res.x
        # Generate a random geometric hash map for spatial reconfiguration
        hash_map = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_map[i, 0]
            perturbed_v[3*i+1] += hash_map[i, 1]
        # Re-optimize with perturbed positions
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Shake heuristic: perturb smallest circles to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        small_radius_indices = np.argsort(radii)[:5]
        for idx in small_radius_indices:
            perturbation = np.random.uniform(-0.02, 0.02, size=2)
            v[3*idx] += perturbation[0]
            v[3*idx+1] += perturbation[1]
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted expansion on the smallest non-zero radius while enforcing reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the smallest non-zero radius to expand
        min_radius_idx = np.argmin(radii)
        if radii[min_radius_idx] < 1e-6:
            min_radius_idx = np.argmax(radii)
        total_sum = np.sum(radii)
        # Compute maximum possible expansion while maintaining feasibility
        max_expansion = 0.0
        for i in range(n):
            if i != min_radius_idx:
                # Test expansion of the smallest radius
                expanded_radii = radii.copy()
                expanded_radii[min_radius_idx] += 0.001
                expanded_radii[i] -= 0.001
                # Check if expansion is feasible
                if validate_packing(np.column_stack([v[0::3], v[1::3]]), expanded_radii)[0]:
                    max_expansion = 0.001
                else:
                    break
        # Expand the smallest radius
        expanded_radii = radii.copy()
        expanded_radii[min_radius_idx] += max_expansion
        for i in range(n):
            if i != min_radius_idx:
                expanded_radii[i] -= max_expansion / (n - 1)
        # Re-optimize with new radii
        expanded_v = v.copy()
        expanded_v[2::3] = expanded_radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())