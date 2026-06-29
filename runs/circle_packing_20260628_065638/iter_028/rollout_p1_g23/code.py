import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with non-uniform spatial distribution using adaptive Voronoi tessellation bias
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols * (1.0 + 0.25 * np.sin(0.1*i))
        y_center = (row + 0.5) / rows * (1.0 + 0.25 * np.cos(0.1*i))
        # Randomized offset for non-local spatial disruption
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift rows with adaptive stagger
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 + 0.1 * np.sin(0.1*i))
        xs.append(x)
        ys.append(y)
    
    # Adaptive base radii based on spatial density
    base_area = (1.0 - 1e-2)**2 / n
    r0 = np.sqrt(base_area) - 1e-2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with lambda capture
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with lambda capture for i and j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with adaptive max iterations based on density
    initial_steps = 1000 + 500 * np.sqrt(n)
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": int(initial_steps), "ftol": 1e-12})
    
    # Radical spatial reconfiguration with non-local hashing and radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate non-local geometric hash with directional bias and spatial correlation
        spatial_hash = np.random.rand(n, 2) * 0.08
        # Generate adjacency-aware expansion vectors with spatial correlation
        adj_expansion = np.random.rand(n, 2) * 0.03
        # Apply non-local spatial perturbation with adaptive scaling
        perturbed_v = v.copy()
        for i in range(n):
            # Non-local spatial perturbation
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
            # Directional expansion for adjacency-aware constraint violation
            if i < n - 2:
                if 0.5 < abs(v[3*i+1] - v[3*i+1 + 3]) < 0.8:
                    perturbed_v[3*i+2] += adj_expansion[i, 0] * 0.005
                    perturbed_v[3*i+1] += adj_expansion[i, 1] * 0.003
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Targeted radius expansion using non-local spatial hashing and constraint-aware expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Efficient broadcast distance computation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by minimizing minimum neighbor distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Minimize distance to others
        # Targeted expansion with spatial hashing and directional bias
        new_radii = radii.copy()
        # Boost expansion for the most spatially isolated circle
        new_radii[least_constrained_idx] += 0.008 * (1.5 + 0.3 * np.random.rand())
        # Expand nearby circles with spatial hashing and directional expansion
        for i in range(n):
            if i != least_constrained_idx:
                adj_weight = np.linalg.norm(centers[i] - centers[least_constrained_idx])
                if adj_weight < 0.15:
                    expansion = 0.004 * (1.0 + 0.2 * np.random.rand())
                else:
                    expansion = 0.002 * (1.0 + 0.1 * np.random.rand())
                new_radii[i] += expansion
        
        # Apply expansion with progressive constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Efficient overlap validation with vectorization
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
                # If invalid, decrease expansion slightly
                # Adaptive decrease based on proximity to overlap
                if np.min(dists) < 0.2:
                    new_radii = radii + (new_radii - radii) * 0.97
                else:
                    new_radii = radii + (new_radii - radii) * 0.99
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final re-evaluation with dynamic radius constraint
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())