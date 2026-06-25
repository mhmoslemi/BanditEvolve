import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    for i in range(n):
        for j in range(i + 1, n):
            def constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2
            cons.append({"type": "ineq", "fun": constraint})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Cleanup pass: attempt to slightly increase radii without moving centers
    def try_inflate(radii, centers, bounds, cons):
        def neg_sum_radii_inflate(v):
            return -np.sum(v)
        new_r = radii.copy()
        for i in range(n):
            # Try to increase radius slightly if possible
            max_inflate = 0.0
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    max_inflate = max(max_inflate, min_dist - dist)
            if max_inflate > 0:
                new_r[i] += max_inflate * 0.001
        # Check if the new radii are valid
        valid, msg = validate_packing(centers, new_r)
        if valid:
            return new_r
        else:
            return radii
    
    radii = try_inflate(radii, centers, bounds, cons)
    
    # Apply controlled randomness for probabilistic exploration
    def apply_random_perturbations(radii, centers, bounds, cons):
        # Use a small Gaussian perturbation with standard deviation 0.001
        new_r = radii.copy()
        for i in range(n):
            # Perturb radius slightly
            new_r[i] += np.random.normal(0, 0.001)
            # Clamp to bounds
            new_r[i] = np.clip(new_r[i], bounds[3*i + 2][0], bounds[3*i + 2][1])
        # Check if the new radii are valid
        valid, msg = validate_packing(centers, new_r)
        if valid:
            return new_r
        else:
            return radii
    
    radii = apply_random_perturbations(radii, centers, bounds, cons)
    return centers, radii, float(radii.sum())