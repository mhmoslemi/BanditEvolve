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
    
    # Vectorized overlap constraints
    x = v0[0::3]
    y = v0[1::3]
    r = v0[2::3]
    X, Y = np.meshgrid(x, x)
    Y, Z = np.meshgrid(y, y)
    R = np.outer(r, r)
    
    # Compute squared distances and squared sum of radii
    dx = X - Y
    dy = Z - Y
    dist_sq = dx**2 + dy**2
    r_sum_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
    
    # Flatten indices for constraint creation
    indices = np.triu(np.ones((n, n)), k=1).nonzero()
    for i, j in zip(indices[0], indices[1]):
        cons.append({"type": "ineq", "fun": lambda v, i=i, j=j:
                     (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                     - (v[3*i+2] + v[3*j+2])**2})

    # Add penalty for overlapping circles to improve convergence
    def penalty(v):
        sum_penalty = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                    sum_penalty += max(0, (v[3*i+2] + v[3*j+2] - dist) ** 2)
        return sum_penalty

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Apply 'shake' heuristic to smallest circles
    if radii.min() > 1e-6:
        small_circle_indices = np.argsort(radii) < 5  # Select smallest 5 circles
        v_shake = v.copy()
        for i in np.where(small_circle_indices)[0]:
            # Perturb center slightly
            v_shake[3*i] += np.random.uniform(-0.01, 0.01)
            v_shake[3*i+1] += np.random.uniform(-0.01, 0.01)
            # Adjust radius to maintain feasibility
            r = v_shake[3*i+2]
            v_shake[3*i+2] = np.clip(r - 0.001, 1e-6, 0.5)

        # Re-optimize with perturbed configuration
        res = minimize(neg_sum_radii, v_shake, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
        v = res.x if res.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())