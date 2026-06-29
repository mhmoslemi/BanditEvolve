import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a hexagonal grid pattern with spatial variation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Introduce variation to break symmetry and allow better expansion
        if row % 3 == 1:
            x += 0.1 / cols
        if row % 2 == 1:
            y += 0.1 / rows
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

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Shake heuristic for small circles to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify small circles to shake
        small_circle_indices = np.where(radii < np.quantile(radii, 0.2))[0]
        if len(small_circle_indices) > 0:
            # Randomly select one small circle to perturb
            perturbation_index = small_circle_indices[np.random.randint(len(small_circle_indices))]
            # Small random perturbation to the position
            perturbation = 0.02 * np.random.rand(2)
            v[3*perturbation_index + 0] += perturbation[0]
            v[3*perturbation_index + 1] += perturbation[1]
            # Ensure the perturbed circle stays within bounds
            v[3*perturbation_index + 0] = np.clip(v[3*perturbation_index + 0], 1e-6, 1.0 - 1e-6)
            v[3*perturbation_index + 1] = np.clip(v[3*perturbation_index + 1], 1e-6, 1.0 - 1e-6)
            # Re-optimize with perturbed parameters
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())