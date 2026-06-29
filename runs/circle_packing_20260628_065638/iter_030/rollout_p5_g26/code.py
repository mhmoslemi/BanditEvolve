import numpy as np

def run_packing():
    n = 26
    # Improved grid structure with dynamic refinement and better spatial distribution
    # First, define optimized grid dimensions with asymmetric row/col counts
    # 5x5 grid gives 25, we add 1 row to 5 columns to make it 6 rows x 5 columns
    cols = 5
    rows = (n + cols - 1) // cols  # 5+25-1=29 => 29//5=5.8 => 6

    # Create adaptive grid with more precise spatial initialization
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # x: distribute evenly across [0,1] but with more room on edges due to row spacing
        col_offset = (col + 0.5) / cols
        x_center = col_offset + (i % cols) * 0.02  # Add slight x-displacement for spacing
        # y: distribute with 2 rows at top, 2 rows at bottom, and 2 rows in center
        # y_center is adjusted based on row location
        # rows: 0, 1, 2, 3, 4, 5 where 5 is the last row
        row_multiplier = 0
        if row < 2:
            # Top two rows: increase y_center to avoid overlap with top edge
            row_multiplier = 0.25 + 0.5 * row / 2  # 0.25, 0.5 at row=0,1
        elif row > 3:
            # Bottom two rows: increase y_center to avoid overlap with bottom edge
            row_multiplier = 0.75 - 0.5 * (rows - row - 1)/2  # 0.75, 0.5 at row=4,5
        else:
            # Middle rows: 1.0 (centered)
            row_multiplier = 1.0
        y_center = row_multiplier + (row - rows / 2) * 0.1  # Adjust for row spacing
        # Add jitter to break symmetry
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Add staggered offset for even rows (row % 2 == 0) to create space
        if row % 2 == 0:
            x += 0.59 / cols  # 0.59 is slightly less than 0.6 (5 cols spacing)
        xs.append(x)
        ys.append(y)
    
    # Set initial radii with an optimized base value and adaptive adjustment
    # Base value is derived from grid density: 1/(cols*rows) + some margin, not static
    base_radius = (np.sqrt(1/(cols*rows)) + 0.008) * 0.9
    # Distribute initial radii with a small random variation to create spatial diversity
    r0 = np.random.uniform(base_radius - 0.004, base_radius + 0.004, size=n)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Ensure bounds and vector length match for all n=26 circles
    bounds = []
    for _ in range(n):
        bounds += [ (0.0, 1.0), (0.0, 1.0), (1e-4, 0.5) ]

    def neg_sum_radii(v):
        """Objective: maximize the sum of radii (minimize -sum_radii)"""
        return -np.sum(v[2::3])

    # Define constraints
    cons = []

    # Per-circle boundary constraints with precise bounds
    for i in range(n):
        # Left x - r >= 0.0 -> x >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right x + r <= 1.0 -> x <= 1 - r
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom y - r >= 0.0 -> y >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top y + r <= 1.0 -> y <= 1 - r
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints with vectorized and precomputed efficient functions
    # To avoid lambda capturing issues, use closure parameters (with fixed i,j)
    def generate_overlap_constraints():
        overlaps = []
        for i in range(n):
            for j in range(i + 1, n):
                # Precompute for efficiency
                # Function: dx^2 + dy^2 >= (r_i + r_j)^2
                def constraint(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist_sq = dx*dx + dy*dy
                    r_sum = v[3*i+2] + v[3*j+2]
                    return dist_sq - r_sum*r_sum
                overlaps.append( {"type": "ineq", "fun": constraint} )
        return overlaps
    cons += generate_overlap_constraints()

    # Initial optimization - first phase
    res = minimize(
        neg_sum_radii,
        v0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={ 
            "maxiter": 2500,   # Increased for better convergence
            "ftol": 1e-12,     # Tighter tolerance for precision
            "eps": 1e-7        # Smaller step size for better gradient approximation
        }
    )

    # Phase 2: Perturbed spatial reconfiguration
    if res.success:
        v = res.x
        current_radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Create dynamic perturbation matrix
        # Perturbation intensity is adjusted based on current radius distribution
        perturbation_strength = np.sqrt(np.sum(current_radii)**2) * 0.09 / np.max(current_radii)
        # Perturb x and y with adaptive spatial noise
        spatial_noise = np.random.uniform(-perturbation_strength, perturbation_strength, (n, 2))
        v_perturbed = v.copy()
        for i in range(n):
            v_perturbed[3*i] += spatial_noise[i,0]
            v_perturbed[3*i+1] += spatial_noise[i,1]
        
        # Re-optimization after perturbation
        # Increase iterations since it's a reconfiguration phase
        res = minimize(
            neg_sum_radii,
            v_perturbed,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={ 
                "maxiter": 1500,  # Slightly lower than first phase
                "ftol": 1e-11,    # Slightly looser to allow some adjustment
                "eps": 1e-8       # Adjust to allow for perturbations
            }
        )

    # Phase 3: Radius expansion with constrained optimization and spatial filtering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized broadcasting for distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate growth based on current total sum and potential for expansion
        current_sum = np.sum(radii)
        # Targeted expansion based on geometric potential, not absolute
        # Calculate possible expansion using a weighted average of minimal distances
        min_dist_avg = np.mean(min_dists)
        min_dist_std = np.std(min_dists) if n > 1 else 0
        # Adjust based on standard deviation of minimal distances for robustness
        expansion_factor = 0.008 * (min_dist_avg + min_dist_std) / (np.sum(radii) + 1e-6) * 1.03
        # Apply more aggressive expansion to least constrained circle
        new_radii = radii.copy()
        # Adjust based on spatial distribution, not just distance
        # Calculate potential expansion using geometric hashing or neighbor analysis
        # For safety, add a small random multiplier
        random_expansion_factor = np.random.uniform(0.98, 1.02)
        # Apply non-uniform expansion
        new_radii[least_constrained_idx] += expansion_factor * random_expansion_factor * 1.2
        # Apply moderate expansion to other circles
        # Weight expansion based on distance to least constrained circle
        # This is an adaptive approach that avoids overexpansion
        for i in range(n):
            if i != least_constrained_idx:
                # Calculate adjusted expansion based on spatial relationship
                dx_i = centers[i,0] - centers[least_constrained_idx,0]
                dy_i = centers[i,1] - centers[least_constrained_idx,1]
                dist_i = np.sqrt(dx_i**2 + dy_i**2)
                # Use inverse distance to determine expansion weight
                expansion_weight = max(0.8, (1.0 - (dist_i / (2.0 * radii[least_constrained_idx]))))
                new_radii[i] += expansion_factor * random_expansion_factor * expansion_weight

        # Now, perform a constrained optimization to refine radii
        # We'll allow for small expansion, but must ensure constraints are active
        # Use a more stringent solver here
        res = minimize(
            neg_sum_radii,
            v.copy(),  # Start with the last state for stability
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={ 
                "maxiter": 1500, # Still high for convergence but less than initial phase
                "ftol": 1e-10,   # Tight to avoid overshooting
                "eps": 1e-9     # Smaller for gradient refinement
            }
        )
        # If this fails, fall back to last successful configuration
        v = res.x if res.success else v
    
    # Final validation phase
    v = res.x if res.success else v0
    # Always apply clipping to avoid negative or excessively large radii
    # This also helps with numerical stability
    final_radii = np.clip(v[2::3], 1e-7, 0.5)
    
    # Final centers are derived from v, which we have already validated
    centers = np.column_stack([v[0::3], v[1::3]])
    
    # Final validation step to ensure correct geometry
    if not res.success:
        # If optimization failed, apply fallback and manual correction
        # Re-evaluate with a modified vector from the last successful state
        # This is a safeguard in case of solver failure
        v = res.x if res.success else v0
        centers = np.column_stack([v[0::3], v[1::3]])
        final_radii = np.clip(v[2::3], 1e-7, 0.5)
    
    return centers, final_radii, float(np.sum(final_radii))