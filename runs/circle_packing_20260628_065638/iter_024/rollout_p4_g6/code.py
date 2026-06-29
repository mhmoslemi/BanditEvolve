import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric tiling initialization with low-density clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        # Randomly perturb to avoid symmetry and cluster formation
        x_offset = np.random.uniform(-0.03, 0.03)
        y_offset = np.random.uniform(-0.03, 0.03)
        # Add some vertical bias to stagger circles more uniformly
        if row % 3 == 1:
            y_offset += np.random.uniform(-0.015, 0.015)
        x = x_base + x_offset
        y = y_base + y_offset
        xs.append(x)
        ys.append(y)
    
    # Initialize radii based on uniform spacing
    r0 = 0.3 / cols - 1e-4
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
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-11})
    
    # If optimization was successful, perform post-processing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for constraint analysis
        dists = np.zeros((n, n))
        for i in range(n):
            dx = centers[:, 0] - centers[i, 0]
            dy = centers[:, 1] - centers[i, 1]
            dists[i, :] = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle (smallest minimum distance to other circles)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Find the circle with the largest margin for radius expansion
        max_expansion_idx = np.argmin(radii)
        
        # If the least constrained and maximum expansion circles are different, prioritize the least constrained
        # Otherwise, use the same circle for both
        if least_constrained_idx != max_expansion_idx:
            expansion_circle = least_constrained_idx
        else:
            expansion_circle = max_expansion_idx
        
        # Compute current radii and max radius that can be added without overlap
        current_radius = radii[expansion_circle]
        expansion_radius = 0.0
        for j in range(n):
            if j == expansion_circle:
                continue
            min_dist = dists[expansion_circle, j]
            max_possible = min_dist - radii[j] - current_radius
            if max_possible > 0:
                expansion_radius = max(expansion_radius, max_possible)
        
        # Add a controlled expansion to the least constrained circle
        delta_radius = expansion_radius * 0.9  # Avoid over-expansion
        new_radii = radii.copy()
        new_radii[expansion_circle] += delta_radius
        
        # Apply final radius expansion with careful optimization
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Perform a short optimization run with high precision
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())