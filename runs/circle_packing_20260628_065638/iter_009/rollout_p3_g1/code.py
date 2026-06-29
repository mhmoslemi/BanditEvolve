import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Randomized offset to break symmetry
        x += np.random.uniform(-0.05, 0.05)
        y += np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.3 / cols - 1e-3
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
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Asymmetric reconfiguration: replace spatial constraint function with randomized version
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Randomized spatial constraint function
        def randomized_constraint_func(v, i, j):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            r_i = v[3*i+2]
            r_j = v[3*j+2]
            return dx*dx + dy*dy - (r_i + r_j)**2 + np.random.normal(0, 1e-5)
        # Replace overlap constraints with randomized version
        new_cons = []
        for i in range(n):
            for j in range(i + 1, n):
                new_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: randomized_constraint_func(v, i, j)})
        # Perturb the least constrained circle to trigger new layout
        constraint_violations = np.zeros(n)
        for i in range(n):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            # Boundary constraint violation
            constraint_violations[i] = max(0, r - x, r - (1 - x), r - y, r - (1 - y))
            # Overlap constraint violations
            for j in range(n):
                if i != j:
                    dx = x - centers[0][j]
                    dy = y - centers[1][j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    constraint_violations[i] = max(constraint_violations[i], r + radii[j] - dist)
        # Find the least constrained circle
        least_constrained_idx = np.argmin(constraint_violations)
        # Perturb its position with random offset
        perturbation = 0.05 * np.random.rand(3)
        v[3*least_constrained_idx] += perturbation[0]
        v[3*least_constrained_idx+1] += perturbation[1]
        v[3*least_constrained_idx+2] += perturbation[2]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=new_cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Final refinement step: perturb smallest and boundary circles
    if res.success:
        v = res.x
        radii = v[2::3]
        small_indices = np.argsort(radii)[:5]
        boundary_indices = []
        for i in range(n):
            x = v[3*i]
            y = v[3*i+1]
            r = v[3*i+2]
            if x < r or x > 1 - r or y < r or y > 1 - r:
                boundary_indices.append(i)
        # Combine and deduplicate indices
        perturb_indices = np.unique(np.concatenate((small_indices, boundary_indices)))
        # Apply small random perturbation to their positions
        perturbation = 0.05 * np.random.rand(len(perturb_indices) * 3)
        perturbed_v = v.copy()
        idx = 0
        for i in perturb_indices:
            perturbed_v[3*i] += perturbation[idx]
            perturbed_v[3*i+1] += perturbation[idx+1]
            perturbed_v[3*i+2] += perturbation[idx+2]
            idx += 3
        # Clip radii to ensure they stay within bounds
        perturbed_v[2::3] = np.clip(perturbed_v[2::3], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=new_cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())