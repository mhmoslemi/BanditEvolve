import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add small randomized offset to break symmetry
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Create staggered grid for horizontal spacing
        if row % 2 == 1:
            offset = 0.5 / cols * 0.75  # Reduced stagger to allow more compact packing
            x += np.random.uniform(-offset, offset)
        xs.append(x)
        ys.append(y)
    
    # Set initial radius based on a more compact grid structure
    r0 = 0.33 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Define bounds for all 3*n dimensions
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Initialize constraints with tighter tolerances and vectorization
    cons = []

    # Boundary constraints (vectorized)
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints (vectorized with more compact calculation)
    for i in range(n):
        for j in range(i+1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    # Spatial reconfiguration with directional perturbation and compacted geometry
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create compact perturbation grid with directional bias
        perturb_centers = np.zeros((n, 2))
        for i in range(n):
            row = i // cols
            col = i % cols
            x_offset = np.random.uniform(-0.025, 0.015)  # Smaller range for tighter packing
            y_offset = np.random.uniform(-0.015, 0.015)
            perturb_centers[i, 0] = x_offset * (col + 1) / cols  # Column-weighted
            perturb_centers[i, 1] = y_offset * (row + 1) / rows  # Row-weighted
        
        # Perturb the grid with a more compact and directional approach
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturb_centers[i, 0]
            perturbed_v[3*i+1] += perturb_centers[i, 1]
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-12})

    # Advanced radial expansion with spatial hashing and dynamic constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance computation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Find least constrained circle (maximum of minimal distances)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Create a spatial hashing grid for radial expansion
        spatial_hash = np.random.rand(n, 2) * 0.04
        hash_radii = radii + (spatial_hash * radii).sum(axis=1)
        hash_radii = np.clip(hash_radii, 1e-4, 0.5)

        # Compute radial expansion with directional bias and spatial hashing
        expansion_factor = 0.003 / (n - 1)  # Lower base expansion
        directional_hash = np.random.rand(n, 2) * 0.025  # Fine directional variation
        
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3 + directional_hash[least_constrained_idx, 0] * 0.02
        for i in range(n):
            if i != least_constrained_idx:
                # Radial expansion with directionally-biased weighting
                new_radii[i] += expansion_factor * (1.0 + directional_hash[i, 0] * 0.3)
        
        # Validate expansion with tighter safety threshold
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i+1, n):
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
                # Adaptive reduction based on violation
                violation = 1.0 - (new_radii[i] + new_radii[j]) / (dist + 1e-12)
                reduction = 1.0 - np.clip(violation * 0.6, 0.0, 0.3)
                new_radii = radii + (new_radii - radii) * (1.0 - reduction)
        
        # Final optimization with expanded configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())