import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Dynamic fractal-based placement with asymmetric perturbations
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        
        # Fractal perturbation: recursive scaling with row-dependent asymmetry
        x = x_base + np.sin(row * 0.5 + i * 0.1) * 0.03
        y = y_base + np.cos(row * 0.5 + i * 0.1) * 0.03
        
        # Alternate row staggering for asymmetry
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
    
    # Radical spatial reconfiguration: apply fractal-based perturbations to trigger new layout
    if res.success:
        v = res.x
        # Apply fractal-based perturbations to positions
        perturbation = np.sin(np.linspace(0, 2 * np.pi, n)) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation[i]
            perturbed_v[3*i+1] += perturbation[i]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion: expand the outermost circle while enforcing strict boundaries
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate distance to boundary for all circles
        dists = np.zeros(n)
        for i in range(n):
            x, y, r = centers[0][i], centers[1][i], radii[i]
            dists[i] = min(x - r, 1.0 - x - r, y - r, 1.0 - y - r)
        # Identify the outermost circle (minimum distance to boundary)
        outermost_idx = np.argmin(dists)
        # Expand its radius slightly and adjust its position to maintain feasibility
        v[3*outermost_idx + 2] += 0.003
        v[3*outermost_idx] += 0.005
        v[3*outermost_idx+1] += 0.005
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())