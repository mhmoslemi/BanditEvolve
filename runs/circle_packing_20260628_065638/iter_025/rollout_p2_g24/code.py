import numpy as np

def run_packing():
    n = 26
    cols = 5  # Optimal column count for 26 circles
    
    # Improved geometric initialization with randomized spatial hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / ((n + cols - 1) // cols)
        # Randomized spatial hashing for enhanced randomness
        x = x_center + np.random.uniform(-0.04, 0.04) * (1 / (cols + 1))
        y = y_center + np.random.uniform(-0.04, 0.04) * (1 / ((n + cols - 1) // cols + 1))
        # Staggered grid with dynamic staggering
        if row % 2 == 1:
            x += 0.5 / cols if col < cols - 1 else -0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with improved spacing
    r0 = 0.35 / cols - 1e-4
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for all 3n variables
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create constraints safely and efficiently
    cons = []

    # Boundary constraints for all circles
    for i in range(n):
        # Left wall
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right wall
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom wall
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top wall
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints with vectorized computation
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                    - (v[3*i+2] + v[3*j+2])**2
                )
            })

    # Initial optimization with enhanced settings
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 1500,
            "ftol": 1e-12,
            "gtol": 1e-9,
            "eps": 1e-8
        }
    )

    # Apply stochastic geometric reconfiguration
    if res.success:
        v = res.x
        # Create random spatial hash for perturbation
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]

        # Second optimization to find new configuration
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 400,
                "ftol": 1e-11,
                "eps": 1e-8
            }
        )

    # Targeted radius expansion after successful optimization
    if res.success:
        v = res.x
        # Extract centers and radii
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Vectorized distance matrix calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor with adaptive scaling
        base_expansion = 0.008 / n
        expansion_factor = base_expansion * (1.5 + 0.5 * np.random.rand())
        
        # Create adjusted radii with targeted expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.4
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * 0.7

        # Apply expansion with constraint validation
        successful_expansion = False
        try_count = 0
        while try_count < 10:
            # Create new decision vector with expansion
            v_new = v.copy()
            v_new[2::3] = new_radii
            
            # Create new centers for constraint checking
            new_centers = np.column_stack([v_new[0::3], v_new[1::3]])
            valid = True
            
            # Check for overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dx = new_centers[i, 0] - new_centers[j, 0]
                    dy = new_centers[i, 1] - new_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Check for square boundaries
                valid = all(
                    (0.0 <= new_centers[i, 0] - new_radii[i]) and
                    (new_centers[i, 0] + new_radii[i] <= 1.0) and
                    (0.0 <= new_centers[i, 1] - new_radii[i]) and
                    (new_centers[i, 1] + new_radii[i] <= 1.0)
                    for i in range(n)
                )
                
                if valid:
                    successful_expansion = True
                    v = v_new
                    break
            
            # Reduce expansion slightly if constraints not met
            new_radii = radii + (new_radii - radii) * 0.97
            try_count += 1

        if successful_expansion:
            res = minimize(
                neg_sum_radii,
                v,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={
                    "maxiter": 300,
                    "ftol": 1e-11,
                    "eps": 1e-8
                }
            )

    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())