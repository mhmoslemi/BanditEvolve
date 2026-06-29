import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols * (1 - 1e-2)  # Slight contraction to create space
        y_center = (row + 0.5) / rows * (1 - 1e-2)
        # Randomized offset with controlled bias
        offset = np.random.uniform(-0.05, 0.05)
        x = x_center + offset
        y = y_center + offset
        if row % 2 == 1:
            x += 0.4 / cols  # Larger staggered shift
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with optimized geometric hashing and spatial partitioning
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-12})
    
    # Radical non-local reconfiguration: geometric hashing with random spatial distortion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric hash with random transformation matrix for spatial distortion
        R = np.random.rand(n, 2) * 0.02 - 0.01
        for i in range(n):
            # Apply random rotation and spatial translation to break symmetry
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x = centers[i, 0] + cos_a * R[i, 0] - sin_a * R[i, 1]
            y = centers[i, 1] + sin_a * R[i, 0] + cos_a * R[i, 1]
            v[3*i] = x
            v[3*i+1] = y
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})
    
    # Targeted radius expansion and non-overlap reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate mutual distances using optimized broadcasting
        x_coords = centers[:, 0][:, np.newaxis]
        y_coords = centers[:, 1][:, np.newaxis]
        dx = x_coords - x_coords.T
        dy = y_coords - y_coords.T
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify circle with smallest minimum interaction distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor respecting total-sum constraint and non-overlap
        target_total_sum = np.sum(radii) + 0.009
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Create expansion vector with soft enforcement and spatial awareness
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.1  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Apply expansion with slight variance for spatial diversification
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # Reduce expansion if overlap occurs
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with tight tolerances and spatial constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())