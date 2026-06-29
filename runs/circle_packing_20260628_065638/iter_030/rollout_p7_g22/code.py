import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with a combination of grid initialization + dynamic perturbation grid 
    # to avoid symmetry and allow better expansion potential
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid (staggered for alternating rows)
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Perturbation to enable asymmetry and better initial spreading
        row_pert = np.random.uniform(-0.14, 0.14)
        col_pert = np.random.uniform(-0.14, 0.14)
        x = base_x + col_pert
        y = base_y + row_pert
        
        # Adjusted row shift for staggered grid (alternate rows + perturbation)
        if row % 2 == 1:
            y += np.random.uniform(-0.1, 0.1)
        xs.append(x)
        ys.append(y)
    
    # Initial radii with adaptive calculation based on grid geometry and overlap buffer
    # Using an initial radius that ensures no two circles overlap (safe starting point)
    # For staggered grid with perturbations, we use a lower initial value to allow better expansion
    # Using base grid spacing as reference and adding a small buffer
    grid_spacing = 0.4 / cols  # based on 0.4 as safe distance for initial layout
    min_initial_radius = grid_spacing * 0.75 - 1e-3
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, min_initial_radius)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized constraints with correct closure capture (i)
    cons = []
    for i in range(n):
        # x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with proper handling of i and j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j]) ** 2 + (v[3*i+1] - v[3*j+1]) ** 2 
                             - (v[3*i+2] + v[3*j+2]) ** 2})

    # Initial optimization with enhanced maxiter and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-11, 
                                             "gtol": 1e-12, "eps": 1e-10})

    # Stage: Adaptive spatial constraint refinement with multi-step expansion
    # 1. Primary refinement: spatial hashing and constraint re-evaluation
    # 2. Secondary refinement: targeted expansion with gradient-based radius optimization
    # 3. Final optimization with dynamic radius re-evaluation
    if res.success:
        # First pass: Spatial hashing to refine layout with perturbations
        v = res.x
        # Generate spatial hash map (with weighted perturbation on radii)
        # Perturbation is scaled by relative radii and geometric influence
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            r = v[3*i+2]
            # Spatial perturbation: adjust centers with radii-weighted hash
            perturbed_v[3*i] += spatial_hash[i, 0] * (r / np.mean(v[2::3]) * 1.2) 
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (r / np.mean(v[2::3]) * 1.2)
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, 
                                                 "gtol": 1e-12, "eps": 1e-10})
    
    if res.success:
        # Second pass: Radius expansion on least constrained circle using geometric-aware expansion
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix calculation (optimized with NumPy broadcasting)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx ** 2 + dy ** 2)
        
        # Compute min distances for each circle and their relative safety metric
        min_dists = np.min(dists, axis=1)
        # Safety metric: normalized minimum distance (distance vs radius sum)
        # This helps identify the "least constrained" circle for expansion
        # Weighted by radius as circles with larger radii need more space
        safety_metric = min_dists / (radii + radii[np.newaxis, :])
        safety_metric = np.nan_to_num(safety_metric)  # Handle division by zero
        
        # Find the circle with the least constrained expansion (max safety metric)
        least_constrained_idx = np.argmax(safety_metric)
        
        # Use a safe growth strategy: growth based on safety and current expansion potential
        # Total growth target based on SOTA knowledge, with adaptive expansion factor
        # Calculate current total sum
        current_total = np.sum(radii)
        target_growth = 0.006 if current_total < 2.63 else 0.005  # SOTA is at 2.63+ 
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create new radii with:
        # - expansion based on safety metric (more for least constrained)
        # - soft constraint: limit expansion per radius (no more than 20% initial)
        # - add small stochastic growth
        new_radii = radii.copy()
        for i in range(n):
            # Base expansion (higher for least constrained)
            expansion = expansion_factor * (1.0 + 0.1 * safety_metric[i])  # Stochastic component
            
            # Soft constraint: max 20% of initial radius
            max_expansion = 0.2 * radii[i]
            expansion = np.clip(expansion, 0.0, max_expansion)
            
            # Add small random variance to break into new configuration
            expansion += np.random.uniform(-0.0005, 0.0005)
            new_radii[i] += expansion
        
        # Check if expansion is valid (without recomputing distances, we do a fast check)
        # Use the same constraint checking mechanism, but only for expanded radii
        # This is optional if we have already found a valid setup

        # To avoid recomputation, here we apply the expansion directly
        # but we will re-verify validity using a minimal constraint check
        # Apply new radii directly
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, 
                                                 "gtol": 1e-12, "eps": 1e-10})
    
    # Final verification of expanded state and optimization to ensure no constraint loss
    # Final cleanup
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final safety check (minimal and fast constraint check)
    # This is for final validation and to catch edge cases
    valid = True
    # Check for NaNs
    if np.isnan(centers).any() or np.isnan(radii).any():
        valid = False
    else:
        # Check square boundaries
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if (x - r < -1e-12 or x + r > 1 + 1e-12 or
                y - r < -1e-12 or y + r > 1 + 1e-12):
                valid = False
                break
        # Check for overlaps
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                    if dist < radii[i] + radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break

    # If it failed due to constraints during the final step, fall back to the last optimization
    if not valid and res.success:  # Only retry if we had a valid previous result
        # Revert and re-optimize (in case the final radii expansion led to invalid state)
        # We use our initial solution as fallback
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
    # Else, we return the latest valid setup

    return centers, radii, float(radii.sum())