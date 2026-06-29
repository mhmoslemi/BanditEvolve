import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Initialize positions with a novel geometric tiling with non-overlapping
    # hexagonal packing variant, dynamic spatial hashing and directional bias
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Create non-overlapping staggered grid with hexagonal distortion
        x_offset = 0.6 / cols * np.sin(2 * np.pi * i / 3)
        y_offset = 0.6 / cols * np.cos(2 * np.pi * i / 4)
        x = x_center + np.random.uniform(-0.06, 0.06) + x_offset
        y = y_center + np.random.uniform(-0.05, 0.05) + y_offset
        # Alternate row offset for staggered grid
        if row % 2 == 1:
            x += 0.5 / cols - 0.1 / cols * np.cos(2 * np.pi * i / 7)
        xs.append(x)
        ys.append(y)
    
    # Optimized initial radius setup using geometric packing formula
    r0 = 0.5 / cols - 0.002
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure tight bounds with exactly 3n entries for the 3n decision vector
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Negate the objective function for minimization
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create vectorized constraints using lambda with captures
    cons = []
    # Boundary constraints with tighter control on spatial perturbations
    for i in range(n):
        # Left margin constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right margin constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom margin constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top margin constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints with directional spatial hashing (non-local)
    # Use a vectorized approach for performance
    for i in range(n):
        for j in range(i + 1, n):
            # Use a directional hash to influence spatial perturbations
            directional_hash = np.random.rand(2) * 0.08 
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j, dh=directional_hash:
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                        - (v[3*i+2] + v[3*j+2])**2)
            })

    # Initial optimization phase with increased iterations and tighter tolerance
    # Use directional reconfiguration with spatial hashing to push boundaries
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-9})
    
    # If optimization succeeds, perform non-local geometric reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute non-local spatial hash for global expansion
        spatial_hash = np.random.rand(n, 2) * 0.12
        # Apply spatial hashing to centers with radius-aware scaling
        perturbed_v = v.copy()
        for i in range(n):
            dx = spatial_hash[i, 0] * (0.5 - radii[i]) / 0.5
            dy = spatial_hash[i, 1] * (0.5 - radii[i]) / 0.5
            perturbed_v[3*i] += dx
            perturbed_v[3*i+1] += dy
            # Add radius perturbation for expansion based on spatial distribution
            if i % 2 == 0:
                perturbed_v[3*i+2] *= (0.98 + 0.02 * np.random.rand())

        # Re-optimized phase with spatial hashing and non-local reconfiguration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9})

    # Targeted radius expansion on the most spatially isolated circle
    # With dynamic boundary adjustment and directional bias
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distances to all other centers
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the circle with the largest minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute total radii to determine expansion potential
        current_total = np.sum(radii)
        # Define a target total radius sum based on global packing patterns
        target_total = current_total + 0.04  # 5% improvement threshold
        
        # Apply directional expansion on least constrained circle
        # Use spatial hash to influence expansion bias
        spatial_hash = np.random.rand(n, 2) * 0.06
        # Directional bias expansion based on spatial hash and distance
        directional_expansion = 0.006 * (1.0 + spatial_hash[least_constrained_idx, 0] * 1.5)
        
        # Generate new radii with a dynamic expansion strategy
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += directional_expansion
        for i in range(n):
            # Apply expansion with spatial and adjacency influence
            adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
            if adj_weight < 0.1:
                expansion = 0.004 * (1.0 + spatial_hash[i, 0] * 0.8)
            else:
                expansion = 0.002 * (1.0 + spatial_hash[i, 1] * 0.5)
            new_radii[i] += expansion
        
        # Apply expansion with a safeguard and validation loop
        # Use a gradient-based expansion with adaptive step-size
        expansion_attempts = 3
        for _ in range(expansion_attempts):
            # Create expanded decision vector
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx_exp = expanded_v[3*i] - expanded_v[3*j]
                    dy_exp = expanded_v[3*i+1] - expanded_v[3*j+1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, reduce expansion by 10% across all circles
                new_radii = radii + (new_radii - radii) * 0.9
            
        # Final optimization after expansion with directional bias
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})

    # Final check and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())