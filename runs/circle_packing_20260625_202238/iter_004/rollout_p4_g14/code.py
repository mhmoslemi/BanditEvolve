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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Constraint-based reordering mutation
    constraint_violations = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                constraint_violations[i] += (v[3*i+2] + v[3*j+2] - dist)
                constraint_violations[j] += (v[3*i+2] + v[3*j+2] - dist)

    sorted_indices = np.argsort(constraint_violations)
    permuted_v = np.zeros_like(v)
    for i, idx in enumerate(sorted_indices):
        permuted_v[3*i] = v[3*idx]
        permuted_v[3*i+1] = v[3*idx+1]
        permuted_v[3*i+2] = v[3*idx+2]

    # Re-optimize with permuted initial guess
    res = minimize(neg_sum_radii, permuted_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else permuted_v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Hybrid geometric-reinforcement mutation
    # Apply small geometric distortion
    distortion = 0.05
    distorted_v = np.copy(v)
    for i in range(n):
        # Radial shift
        x, y, r = v[3*i], v[3*i+1], v[3*i+2]
        if r > 1e-6:
            angle = np.random.uniform(0, 2*np.pi)
            dx = r * np.cos(angle) * distortion
            dy = r * np.sin(angle) * distortion
            distorted_v[3*i] = x + dx
            distorted_v[3*i+1] = y + dy
            # Bound check
            distorted_v[3*i] = np.clip(distorted_v[3*i], 0.0, 1.0)
            distorted_v[3*i+1] = np.clip(distorted_v[3*i+1], 0.0, 1.0)

    # Re-optimize with distorted initial guess
    res = minimize(neg_sum_radii, distorted_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    v = res.x if res.success else distorted_v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Reinforcement mutation: adjust last modified elements
    if np.sum(radii) > 0:
        # Reinforce the last 5 modified circles
        for i in range(n):
            for j in range(i+1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                    # Adjust radii to enforce spacing
                    r1 = v[3*i+2]
                    r2 = v[3*j+2]
                    required_dist = r1 + r2
                    if dist < required_dist - 1e-5:
                        adjustment = (required_dist - dist) * 0.5
                        v[3*i+2] = np.clip(r1 + adjustment, 1e-6, 0.5)
                        v[3*j+2] = np.clip(r2 + adjustment, 1e-6, 0.5)
                        # Update centers if needed
                        if v[3*i+2] > r1:
                            v[3*i] = np.clip(v[3*i] + (v[3*i+2] - r1) * 0.5, 0.0, 1.0)
                        if v[3*j+2] > r2:
                            v[3*j] = np.clip(v[3*j] + (v[3*j+2] - r2) * 0.5, 0.0, 1.0)
                        v[3*i+1] = np.clip(v[3*i+1] + (v[3*i+2] - r1) * 0.5, 0.0, 1.0)
                        v[3*j+1] = np.clip(v[3*j+1] + (v[3*j+2] - r2) * 0.5, 0.0, 1.0)

    return centers, radii, float(radii.sum())