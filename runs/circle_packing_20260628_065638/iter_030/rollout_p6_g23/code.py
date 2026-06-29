import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))  # 5 or 6, more adaptive than fixed 5
    rows = (n + cols - 1) // cols
    
    # Initialize with a more diverse geometric initialization
    xs = []
    ys = []
    seed_indices = np.random.permutation(n)
    for i in range(n):
        row = seed_indices[i] // cols
        col = seed_indices[i] % cols
        x_center_base = (col + 0.25) / cols
        y_center_base = (row + 0.25) / rows
        # Use more adaptive offset based on distance to edge
        margin = (1.0 - 2.0 * (col + 0.25)/cols) ** 0.5  # edge-aware scaling
        offset_radius = 0.04 * (0.8 + 0.2 * np.random.rand())
        x = x_center_base + np.random.uniform(-margin, margin)
        y = y_center_base + np.random.uniform(-margin, margin)
        if row % 2 == 1:  # Alternate row
            x += 0.35 / cols * (1.0 - np.random.rand() * 0.7)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.30 / cols - 1e-3  # Reduce initial radius for more expansion
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3n matches

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint setup: use closures with i and precompute 
    # the constraints with lambda capturing i explicitly
    cons = []

    # Precompute boundary constraints with i-aware closures
    for i in range(n):
        def _get_boundary_constraint(i):
            def fun(v):
                x = v[3*i]
                y = v[3*i + 1]
                r = v[3*i + 2]
                # Left constraint: x - r >= 0
                left_cons = x - r
                # Right constraint: 1 - x - r >= 0
                right_cons = 1.0 - x - r
                # Bottom: y - r >= 0
                bottom_cons = y - r
                # Top: 1 - y - r >= 0
                top_cons = 1.0 - y - r
                return [left_cons, right_cons, bottom_cons, top_cons]
            return fun
        # These constraints need to be handled as separate
        cons.append({"type": "ineq", "fun": _get_boundary_constraint(i)})
        cons.append({"type": "ineq", "fun": _get_boundary_constraint(i)})
        cons.append({"type": "ineq", "fun": _get_boundary_constraint(i)})
        cons.append({"type": "ineq", "fun": _get_boundary_constraint(i)})
    
    # Generate all pairwise overlap constraints using vectorized distance matrices (with broadcasting)
    # Vectorization for overlap constraints is achieved via pre-stored indices
    # We'll generate a grid of pairwise combinations once
    overlap_pairs = np.zeros((n*n, 2), dtype=np.int64)
    for idx in range(n):
        for jdx in range(n):
            if idx < jdx:
                overlap_pairs[idx*n + jdx, 0] = idx
                overlap_pairs[idx*n + jdx, 1] = jdx
    # Prepare these pairs for constraint generation
    overlap_pairs = overlap_pairs[:n*(n-1)//2]  # Only upper triangular
    # Now build constraints
    for a, b in overlap_pairs:
        def _get_overlap_func(a, b):
            def constraint(v):
                dx = v[3*a] - v[3*b]
                dy = v[3*a+1] - v[3*b+1]
                dist_squared = dx*dx + dy*dy
                # We want dist >= r_a + r_b => dist^2 >= (r_a + r_b)^2
                # So use constraint: dist_squared - (r_a + r_b)^2 >= 0
                r_a = v[3*a+2]
                r_b = v[3*b+2]
                return dist_squared - (r_a + r_b) ** 2
            return constraint
        cons.append({"type": "ineq", "fun": _get_overlap_func(a, b)})
    
    # Initial optimization with tighter tolerances and adaptive max iterations
    # We apply a multi-stage strategy: initial, intermediate, and aggressive optimization
    # with varying perturbation and constraint enforcement
    def stagewise_minimize():
        initial_res = minimize(neg_sum_radii, v0, method="SLSQP", 
                            bounds=bounds, constraints=cons,
                            options={"maxiter": 400, "ftol": 1e-10, "eps": 1e-7})
        
        if not initial_res.success:
            # Reinitialization with more diversity in starting positions
            perturbed_v = v0.copy()
            for i in range(n):
                perturbed_v[3*i] += np.random.uniform(-0.04, 0.04)
                perturbed_v[3*i+1] += np.random.uniform(-0.04, 0.04)
                perturbed_v[3*i+2] += np.random.uniform(-0.005, 0.005)
            intermediate_res = minimize(neg_sum_radii, perturbed_v, 
                                     method="SLSQP", bounds=bounds, 
                                     constraints=cons,
                                     options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-8})
            
            if not intermediate_res.success:
                # If all else, apply more aggressive spatial shaking
                perturbed_v = v0.copy()
                for i in range(n):
                    perturbed_v[3*i] += np.random.normal(0, 0.03)
                    perturbed_v[3*i+1] += np.random.normal(0, 0.03)
                    perturbed_v[3*i+2] += np.random.normal(0, 0.003)
                final_res = minimize(neg_sum_radii, perturbed_v, 
                                 method="SLSQP", bounds=bounds, 
                                 constraints=cons,
                                 options={"maxiter": 800, "ftol": 1e-11, "eps": 1e-9})
                return final_res
            
            return intermediate_res
        
        return initial_res
    
    res = stagewise_minimize()
    
    # Post-optimization refinement: adaptive expansion and topological reconfiguration
    # Step 1: Identify most isolated (non-overlapping) candidates
    if res.success:
        v = res.x
        # Compute pairwise distances vectorized using broadcasting
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.power(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2), 0.5)
        # Minimum distance to others for each circle
        min_dists = np.min(dists, axis=1)
        # Index of most isolated circle (least distance to others)
        isolated_idx = np.argmax(min_dists)
        # Find the second isolate and other candidates for topological shift
        # Create a secondary isolation index via minimum mutual min distances
        # This is to avoid total clusters
        secondary_isolated_indices = np.argsort(min_dists)[-3:]  # Top 3 isolated including the main one
        secondary_isolated_indices = np.unique(secondary_isolated_indices)
        
        # Step 2: Radius expansion with topological reordering (swapping)
        # First, create a modified version of the current configuration
        # We will apply a small, controlled expansion to the isolated circles
        # and apply a spatial topological swap to enhance utilization
        v_new = v.copy()
        # Create a radius expansion vector
        expansion_factor = 0.008 / (n - 1) * (0.85 + 0.15 * np.random.rand())  # adaptive expansion rate
        # Create an expansion vector
        expanded_radii = v_new[2::3].copy()
        # Apply expansion to isolated candidates
        for idx in secondary_isolated_indices:
            # Apply more aggressive expansion to most isolated
            # and some randomness to encourage configuration variety
            expanded_radii[idx] += expansion_factor * (1.15 + 0.05*np.random.rand())
        # Apply moderate expansion to others
        for idx in range(n):
            if idx not in secondary_isolated_indices:
                expanded_radii[idx] += expansion_factor * (0.90 + 0.05*np.random.rand())
        
        # Reconfigure centers with a subtle spatial swap (swap two circles' positions)
        # This is a more aggressive reconfiguration to break clusterings
        # Select two distinct circles and swap their positions
        if n > 2:
            # Randomly select two indices
            swap_indices = np.random.choice(n, size=2, replace=False)
            # Swap their positions
            v_new[3*swap_indices[0]], v_new[3*swap_indices[1]] = v_new[3*swap_indices[1]], v_new[3*swap_indices[0]]
            v_new[3*swap_indices[0]+1], v_new[3*swap_indices[1]+1] = v_new[3*swap_indices[1]+1], v_new[3*swap_indices[0]+1]
        
        # Re-evaluate the expanded and reconfigured configuration
        final_res = minimize(neg_sum_radii, v_new, 
                         method="SLSQP", bounds=bounds, 
                         constraints=cons,
                         options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9})
        
        if final_res.success:
            v = final_res.x
        else:
            # Fallback: maintain the current configuration with small perturbations
            v = v.copy()
            for idx in range(n):
                v[3*idx] += np.random.uniform(-0.01, 0.01)
                v[3*idx+1] += np.random.uniform(-0.01, 0.01)
                v[3*idx+2] += np.random.uniform(-0.002, 0.002)
            final_res = minimize(neg_sum_radii, v, 
                             method="SLSQP", bounds=bounds, 
                             constraints=cons,
                             options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-8})
        
        # Apply a final pass of constraint validation
        # We also perform a soft validation to ensure no overlap
        v = final_res.x if final_res.success else v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Perform a final constraint check with relaxed tolerance
        # (in case of some precision issues)
        # We also clip to avoid NaNs and negative radii
        radii = np.clip(radii, 1e-6, 0.5)  # Ensure radii are positive and within unit
        return centers, radii, float(radii.sum())
    
    return v0[0::3], v0[1::3], float(v0[2::3].sum())