import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initial positions: staggered grid with randomized offsets
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
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

    # Vectorized constraint builder
    def build_boundary_constraints(n):
        cons = []
        for i in range(n):
            cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        return cons
    
    # Vectorized constraint builder with proper closure
    def build_overlap_constraints(n):
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                cons.append({"type": "ineq", "fun": constraint_func})
        return cons

    # Initial optimization
    constraints = build_boundary_constraints(n) + build_overlap_constraints(n)
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Shake heuristic: perturb small circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Find the smallest circles (excluding the smallest to avoid over-expansion)
        small_indices = np.argsort(radii)[:2]
        perturb_strength = 0.015  # Controlled perturbation intensity
        perturbed_v = v.copy()
        for idx in small_indices:
            delta_x = np.random.uniform(-perturb_strength, perturb_strength)
            delta_y = np.random.uniform(-perturb_strength, perturb_strength)
            perturbed_v[3*idx] += delta_x
            perturbed_v[3*idx+1] += delta_y
        
        # Re-optimization with perturbations
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 600, "ftol": 1e-11})
    
    # Optimization with soft expansion factor and vectorized distance calculations
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle (max minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Apply controlled expansion: increase radii of least constrained circle
        expansion_factor = 0.006 / (n - 1)
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3
        
        # Re-evaluate with new radii
        v_expanded = v.copy()
        v_expanded[2::3] = new_radii
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())