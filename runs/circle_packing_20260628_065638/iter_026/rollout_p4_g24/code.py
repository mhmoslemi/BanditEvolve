import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Random initialization with advanced geometric distribution
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid with staggered rows and randomization
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Add geometric randomness with spatial-aware variance
        x_noise = np.random.normal(0, 0.06, 1)
        y_noise = np.random.normal(0, 0.06, 1)
        
        # Alternate row staggering for dense packing
        if row % 2 == 1:
            base_x += 0.5 / cols
        
        x = base_x + x_noise[0]
        y = base_y + y_noise[0]
        
        # Add directional randomization based on quadrant
        if col % 3 == 0:
            x += np.random.uniform(-0.03, 0.03)
        elif col % 3 == 1:
            y += np.random.uniform(-0.03, 0.03)
        elif col % 3 == 2:
            x -= np.random.uniform(-0.03, 0.03)
        
        xs.append(np.clip(x, 0.0001, 0.9999))
        ys.append(np.clip(y, 0.0001, 0.9999))
    
    # Initialize radii with geometric clustering
    r0 = 0.28 / cols
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.4)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with advanced boundary handling
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with adaptive hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "maxls": 100})

    # Disruptive geometric hashing for reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial hash for stochastic redistribution
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        
        # Apply spatial hashing with adaptive scaling
        for i in range(n):
            scale = 1.0 + 0.1 * np.random.rand()
            perturbed_v[3*i] += scale * spatial_hash[i, 0]
            perturbed_v[3*i+1] += scale * spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "maxls": 100})

    # Targeted radius expansion with geometric enforcement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix for constrained expansion
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify circle with smallest non-zero radius (least constrained)
        min_radii = np.min(radii)
        isolated_idx = np.argmin(radii)
        
        # Calculate expansion factor with topological awareness
        target_total_sum = np.sum(radii) + 0.01
        expansion = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Apply expansion with strict non-overlap enforcement
        new_radii = radii.copy()
        new_radii[isolated_idx] += expansion * 1.1
        
        # Apply expansion to others with spatial awareness
        for i in range(n):
            if i != isolated_idx:
                # Calculate direction vector to nearest neighbor
                neighbors = np.argsort(dists[i, :])
                nearest = neighbors[0]
                
                # Adjust expansion based on spatial context
                if i % 2 == 0 and nearest % 2 == 1:
                    expansion_i = expansion * (1.0 + 0.05 * np.random.rand())
                else:
                    expansion_i = expansion * (1.0 - 0.03 * np.random.rand())
                
                new_radii[i] += expansion_i
        
        # Create new decision vector for constrained optimization
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "maxls": 100})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())