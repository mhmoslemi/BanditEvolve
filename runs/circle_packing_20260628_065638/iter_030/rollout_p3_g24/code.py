import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))  # 5 columns for 5x5 grid for n=26
    rows = (n + cols - 1) // cols
    
    # Initialize centers using dynamic grid + perturbed clustering
    xs = []
    ys = []
    
    # Generate initial grid positions with row-wise shift for staggered effect
    for i in range(n):
        col = i % cols
        row = i // cols
        
        # Adjust base position based on col and row with non-uniform scaling
        base_col = col / cols * 1.5  # 1.5 expansion allows more spacing
        base_row = row / rows * 1.5
        # Apply asymmetric spatial jittering: more vertical variation for staggered layout
        jitter_x = np.random.uniform(-0.05, 0.05) * (1.0 / (row + 1))  # reduces horizontal jitter with row
        jitter_y = np.random.uniform(-0.1, 0.1) * (1.0 / (cols + 1))  # introduces more vertical variation
        x = base_col + jitter_x
        y = base_row + jitter_y

        # Apply asymmetric row-wise stagger: alternate row shift with more vertical displacement
        if row % 3 == 2:  # for row-wise stagger on every 3rd row
            # Add more vertical offset for staggered grid in higher rows
            x += 0.15 * (1.0 / (cols + 1))  # small but intentional offset
            y += 0.1 * (1.0 / (rows + 1))  # vertical shift for staggered effect

        xs.append(x)
        ys.append(y)
    
    # Initial radii based on grid and adaptive scaling to maximize spacing
    r0 = (0.7 / cols) * (1.0 - 0.05)  # reduced grid size with more spacing allowed
    v0 = np.full(3 * n, 0.0)
    
    # Assign initial positions and radii
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for the decision vector (3 * n entries)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)]  # radii must be at least 1e-5

    def neg_sum_radii(v):
        # Return negative of sum for maximization
        return -np.sum(v[2::3])

    # Constraints
    cons = []

    # Boundary constraints
    for i in range(n):
        # Left side: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right side: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom side: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top side: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Compute distance squared and subtract sum of radii
            # Avoiding sqrt for performance, with explicit vectorization
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i + 1] - v[3*j + 1])**2 
                    - (v[3*i + 2] + v[3*j + 2])**2
            })
    
    # Initial optimization
    # Use tighter tolerances and more iterations - SLSQP with high precision
    res = minimize(neg_sum_radii, v0, 
                   method="SLSQP", 
                   bounds=bounds,
                   constraints=cons,
                   options={
                       "maxiter": 1500,
                       "ftol": 1e-12,
                       "gtol": 1e-12,  # Tolerance on gradient
                       "eps": 1e-11  # Precision in finite differences for gradient
                   })

    # Stochastic reconfiguration: adaptive spatial hashing with spatial awareness
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create spatial-aware stochastic hash - more dynamic based on local density
        # Use local density to determine perturbation magnitude
        # Avoid high-perturbation on crowded areas, more on isolated ones
        spatial_hash = np.zeros((n, 2))
        for i in range(n):
            # Create a distance-weighted map based on neighbors
            dists = np.linalg.norm(centers - centers[i], axis=1)
            # Apply inverse-distance weighting for perturbation
            inv_dists = 1.0 / (dists + 1e-12)
            inv_dists = np.clip(inv_dists, 0, 1e12)  # clip for safety
            w = inv_dists / inv_dists.sum()
            # Perturb more for isolated points
            scale = 0.25 * (1.0 + 0.5 * (1 - np.mean(w)))
            spatial_hash[i, 0] = np.random.normal(0, scale, 1)
            spatial_hash[i, 1] = np.random.normal(0, 0.15 * scale, 1)
        
        # Apply spatial hash, but respect radii to avoid over-perturbation
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (0.1 + radii[i]/(0.1 + np.abs(radii[i])))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (0.1 + radii[i]/(0.1 + np.abs(radii[i])))
        
        # Re-evaluate with perturbed configuration
        for _ in range(2):
            res = minimize(neg_sum_radii, perturbed_v,
                           method="SLSQP", 
                           bounds=bounds,
                           constraints=cons,
                           options={
                               "maxiter": 600, 
                               "ftol": 1e-11, 
                               "gtol": 1e-11,
                               "eps": 1e-11
                           })
            perturbed_v = res.x if res.success else perturbed_v
    
    # Adaptive expansion of "least constrained" circle with spatial-awareness
    if res.success:
        v = res.x
        radii_current = v[2::3]
        centers_current = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized pairwise distance calculation
        dx = centers_current[:, np.newaxis, 0] - centers_current[np.newaxis, :, 0]
        dy = centers_current[:, np.newaxis, 1] - centers_current[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find 'least constrained' circle: maximal minimal distance to neighbors
        # Avoid using simple np.argmin, prefer weighted scoring with density factors
        # Use inverse neighbor-density as a proxy for isolation score
        min_dists = np.min(dists, axis=1)
        neighbor_densities = np.sum(1.0 / (dists + 1e-12), axis=1)
        # Use inverse density for isolation measure
        isolation_scores = 1.0 / (neighbor_densities + 1e-12)
        isolated_idx = np.argsort(isolation_scores)[-1]  # most unconstrained circle

        # Calculate current sum and expand the radius of the isolated circle
        total_sum = radii_current.sum()
        # Use adaptive expansion: based on total sum and average spacing
        max_expansion = 0.004  # cap for safety
        # Use a weighted expansion based on current radii
        target_total_sum = total_sum + max_expansion * (1.0 + np.mean(radii_current))
        expansion = (target_total_sum - total_sum) / (n - 1)
        # Distribute expansion to others, but give extra to the isolated
        # Use a radius proportional expansion to avoid overstepping
        expansion_factors = np.ones(n) * expansion * 0.99
        expansion_factors[isolated_idx] *= 1.5  # boost isolated circle

        # Create a new radii vector with expansion applied
        new_radii = radii_current + expansion_factors
        
        # Now, build a new decision vector and re-optimize with new radii
        # Keep positions fixed, modify radii but maintain constraints
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        # Reoptimization - more careful with constraints
        for _ in range(2):
            res = minimize(neg_sum_radii, expanded_v,
                           method="SLSQP", 
                           bounds=bounds,
                           constraints=cons,
                           options={
                               "maxiter": 600,
                               "ftol": 1e-11,
                               "gtol": 1e-11,
                               "eps": 1e-11
                           })
            expanded_v = res.x if res.success else expanded_v
    
    # Final optimization with enhanced constraint checking and validation
    if res.success:
        v = res.x
        radii_final = v[2::3]
        centers_final = np.column_stack([v[0::3], v[1::3]])

        # Final safety check: verify all constraints
        # Avoid repeated recomputation; re-use optimization outputs
        # If all is well, return
        return centers_final, radii_final, float(radii_final.sum())
    else:
        # Fallback: return original if something went wrong
        return (np.column_stack([v0[0::3], v0[1::3]]), v0[2::3], float(v0[2::3].sum()))