import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize with a hybrid geometric and stochastic perturbation pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Base coordinates with jitter for random dispersion
        x = base_x + np.random.uniform(-0.08, 0.08)
        y = base_y + np.random.uniform(-0.08, 0.08)
        # Alternate row staggering
        if row % 2 == 1:
            x += 0.5 / cols * 0.9
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

    # Construct constraints with functional capture and vectorization
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with tight constraints and max iter
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-12})

    # Radically reconfigure with adaptive spatial displacement hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash with adaptive displacement
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with enhanced perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 700, "ftol": 1e-11, "eps": 1e-12})
    
    # Implement topological reordering and radial expansion on smallest circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation with optimized broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify smallest radius circle
        smallest_idx = np.argmin(radii)
        smallest_radius = radii[smallest_idx]
        smallest_center = centers[smallest_idx]
        
        # Compute adaptive expansion factor based on cluster density and spacing
        cluster_density = np.sum(dists < radii[:, np.newaxis] + radii[np.newaxis, :], axis=1)
        expansion_factor = (0.0055) * (1.0 + 0.6 * (1 - cluster_density[smallest_idx] / (n-1)))

        # Initial radial expansion with constraint-aware refinement
        new_radii = radii.copy()
        new_radii[smallest_idx] += expansion_factor * 1.3
        for i in range(n):
            if i != smallest_idx:
                # Introduce stochasticity in expansion to explore new regions
                new_radii[i] += expansion_factor * (0.8 + 0.1 * np.random.rand())
        
        # Perform multiple rounds of validation/adjustment
        for _ in range(3):
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
                # Reduce expansion slightly if overlaps detected
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector with refined radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate final configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())