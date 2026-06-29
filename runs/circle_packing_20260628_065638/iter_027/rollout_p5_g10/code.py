import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with structured grid with randomized spatial hashing and dynamic adjustment
    xs = []
    ys = []
    random_offset = 0.06
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-random_offset, random_offset)
        y = y_center + np.random.uniform(-random_offset, random_offset)
        # Stagger rows for better spacing
        if row % 2 == 1:
            x += 0.4 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.36 / cols - 1e-2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with lambda closures to avoid closures capturing variables
    cons = []
    for i in range(n):
        # Left: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right: x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top: y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})

    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2,)})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})

    if res.success:
        # Apply targeted geometric hashing reconfiguration with spatial hashing and dynamic spatial constraint shifting
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Random spatial hashing to simulate new spatial configuration
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluated with new spatial hash
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Vectorized distance calculation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Identify least constrained circle with spatial hashing and adjacency weighting
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)
        
        # Create spatial constraint-aware radius expansion with directional bias
        # Introduce new adjacency-based constraint: radius expansion proportional to spatial adjacency
        # Create adjacency matrix based on normalized spatial proximity
        adjacency_weight = np.zeros(n)
        for i in range(n):
            if i != least_constrained_idx:
                dist = np.linalg.norm(centers[i] - centers[least_constrained_idx])
                if dist < 0.05:
                    adjacency_weight[i] = 1.0
                else:
                    adjacency_weight[i] = 0.5
        adjacency_weight[least_constrained_idx] = 0.0  # Least constrained circle is not adjacent to itself
        
        expansion_factor = 0.003 * (np.sum(adjacency_weight) / np.sum(adjacency_weight + 1e-9))
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Over-expand slightly
        for i in range(n):
            if i != least_constrained_idx:
                # Apply directional expansion based on adjacency
                if adjacency_weight[i] > 0.0:
                    new_radii[i] += expansion_factor * adjacency_weight[i] * (np.random.rand() * 0.3 + 0.6)
                else:
                    new_radii[i] += expansion_factor * (np.random.rand() * 0.15 + 0.05)
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate against all other circles
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, scale down expansion
                new_radii = radii + (new_radii - radii) * 0.96

        # Final optimization with expanded radii and new configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())