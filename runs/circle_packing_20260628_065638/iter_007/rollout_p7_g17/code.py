import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using Voronoi tessellation-inspired placement with more variation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        
        # Introduce more variation to break symmetry and allow better expansion
        if row % 3 == 1:
            x += 0.15 / cols
        if row % 2 == 1:
            y += 0.15 / rows
        
        # Add small random perturbation to avoid symmetry
        x += np.random.uniform(-0.02, 0.02)
        y += np.random.uniform(-0.02, 0.02)
        
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

    # Vectorized constraint setup for efficiency
    cons = []
    for i in range(n):
        # Boundary constraints for x
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Boundary constraints for y
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
    
    # Shake heuristic: perturb the smallest circles and re-optimize
    if res.success:
        v = res.x
        radii = v[2::3]
        # Select the smallest 8 circles for perturbation
        smallest_indices = np.argsort(radii)[:8]
        # Perturb their positions and radii
        for idx in smallest_indices:
            perturbation = np.random.uniform(-0.01, 0.01, size=3)
            v[3*idx + 0] += perturbation[0]
            v[3*idx + 1] += perturbation[1]
            v[3*idx + 2] += perturbation[2]
            # Clip to bounds
            v[3*idx + 0] = np.clip(v[3*idx + 0], 0.0, 1.0)
            v[3*idx + 1] = np.clip(v[3*idx + 1], 0.0, 1.0)
            v[3*idx + 2] = np.clip(v[3*idx + 2], 1e-4, 0.5)
        # Re-optimize with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    # Final optimization
    if res.success:
        v = res.x
        radii = v[2::3]
        # Select the smallest 4 circles again for a final tweak
        smallest_indices = np.argsort(radii)[:4]
        for idx in smallest_indices:
            perturbation = np.random.uniform(-0.005, 0.005, size=3)
            v[3*idx + 0] += perturbation[0]
            v[3*idx + 1] += perturbation[1]
            v[3*idx + 2] += perturbation[2]
            v[3*idx + 0] = np.clip(v[3*idx + 0], 0.0, 1.0)
            v[3*idx + 1] = np.clip(v[3*idx + 1], 0.0, 1.0)
            v[3*idx + 2] = np.clip(v[3*idx + 2], 1e-4, 0.5)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())