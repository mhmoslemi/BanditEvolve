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

    # Add penalty to the objective function
    def neg_sum_radii_with_penalty(v):
        return neg_sum_radii(v) + 1e-3 * penalty(v)

    res = minimize(neg_sum_radii_with_penalty, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Shake the smallest circles to escape local minima
    if res.success and np.sum(radii) > 0:
        small_indices = np.argsort(radii)[:5]
        for idx in small_indices:
            perturbation = np.random.uniform(-0.01, 0.01, size=2)
            v[3*idx] += perturbation[0]
            v[3*idx+1] += perturbation[1]
            # Re-evaluate constraints after perturbation
            # This is a heuristic approach to avoid re-optimizing from scratch
            # and is not fully rigorous but helps in practice
            
            # Ensure the perturbed circle remains within bounds
            v[3*idx] = np.clip(v[3*idx], 0.0, 1.0)
            v[3*idx+1] = np.clip(v[3*idx+1], 0.0, 1.0)
            v[3*idx+2] = np.clip(v[3*idx+2], 1e-6, 0.5)
        
        # Re-optimize slightly with perturbed values
        res = minimize(neg_sum_radii_with_penalty, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 100, "ftol": 1e-9})
        v = res.x if res.success else v
    
    return centers, radii, float(radii.sum())