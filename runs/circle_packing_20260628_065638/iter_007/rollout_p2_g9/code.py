import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using Voronoi tessellation-inspired placement with enhanced randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Introduce variation to break symmetry and allow better expansion
        if row % 3 == 1:
            x += np.random.uniform(-0.05 / cols, 0.05 / cols)
        if row % 2 == 1:
            y += np.random.uniform(-0.05 / rows, 0.05 / rows)
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

    # Precompute all pairwise indices for overlap constraints
    pair_indices = []
    for i in range(n):
        for j in range(i+1, n):
            pair_indices.append((3*i, 3*i+1, 3*j, 3*j+1))
    
    cons = []
    # Add boundary constraints
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, a, b: v[a] - v[b], "args": (3*i, 3*i+2)})
        cons.append({"type": "ineq", "fun": lambda v, a, b: 1.0 - v[a] - v[b], "args": (3*i, 3*i+2)})
        cons.append({"type": "ineq", "fun": lambda v, a, b: v[a] - v[b], "args": (3*i+1, 3*i+2)})
        cons.append({"type": "ineq", "fun": lambda v, a, b: 1.0 - v[a] - v[b], "args": (3*i+1, 3*i+2)})
    
    # Add overlap constraints
    for a, b, c, d in pair_indices:
        def constraint_func(v, a=a, b=b, c=c, d=d):
            dx = v[a] - v[c]
            dy = v[b] - v[d]
            return dx*dx + dy*dy - (v[a+2] + v[c+2])**2
        cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Shake heuristic: perturb the smallest circles and re-optimizing to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        # Select the smallest 10 circles for perturbation
        small_indices = np.argsort(radii)[:10]
        # Apply small random perturbation to their positions
        perturbation = 0.05 * np.random.rand(len(small_indices) * 3)
        perturbed_v = v.copy()
        idx = 0
        for i in small_indices:
            perturbed_v[3*i] += perturbation[idx]
            perturbed_v[3*i+1] += perturbation[idx+1]
            perturbed_v[3*i+2] += perturbation[idx+2]
            idx += 3
        # Clip radii to ensure they stay within bounds
        perturbed_v[2::3] = np.clip(perturbed_v[2::3], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())