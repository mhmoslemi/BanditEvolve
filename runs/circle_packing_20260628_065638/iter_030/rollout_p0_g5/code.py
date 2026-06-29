import numpy as np

def run_packing():
    n = 26
    cols = 6  # Adjusted grid cols to 6 for balanced layout, more flexibility in radial expansion
    rows = (n + cols - 1) // cols
    max_radius_start = 0.5 / (cols * np.sqrt(2)) * 1.2  # Increased starting for better convergence
    
    # Seed for consistent optimization process
    seed = 123456789
    np.random.seed(seed)
    
    # Dynamic grid allocation with initial positions
    xs = []
    ys = []
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        x_center = ((col_idx + 0.3) / cols)  # Slight shift to increase spacing toward edges
        y_center = ((row_idx + 0.35) / rows)  # Slight shift for balanced y positioning
        # Add perturbation to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Staggered row shifting for better hexagonal grid-like distribution
        shift = 0.5 / (cols * 2)
        if row_idx % 2 == 1:
            x += shift * (1 - (col_idx / (cols-1)))  # Row shift increases with column index in offset rows
        xs.append(x)
        ys.append(y)
    
    # Initial radius allocation with gradient ascent potential
    r0 = max_radius_start * np.random.rand(n)  # Randomized starting radii for more diverse solution space
    r0 = np.clip(r0, 1e-4, 0.45)  # Clip to avoid early optimization issues
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    bounds = []
    # Ensure bounds are 3n in length, each circle has its own boundary
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # x, y, radius each have bounds

    def neg_sum_radii(v):
        radii = v[2::3]
        return -np.sum(radii)

    # Define constraints with explicit bounds + spacing constraints
    cons = []
    for i in range(n):
        # Left side constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right side constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized non-overlap constraints (distance >= sum of radii)
    for i in range(n):
        for j in range(i+1, n):
            def distance_constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": distance_constraint_func})
    
    # Optimization parameters
    # Use a more modern set of settings for convergence: hybrid approach with multiple phases
    phase_options = [
        {"method": "SLSQP", "maxiter": 300, "ftol": 1e-12, "eps": 1e-2, "jac": None},  # Phase 1: rough optimization
        {"method": "L-BFGS-B", "maxiter": 600, "ftol": 1e-11, "gtol": 1e-8, "eps": 2e-3},  # Phase 2: finer adjustment
        {"method": "SLSQP", "maxiter": 400, "ftol": 1e-13, "eps": 1e-3, "jac": None}  # Phase 3: precise final optimization
    ]
    
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, constraints=cons, 
                   options={"maxiter": 500, "ftol": 1e-12})
    
    # Advanced constraint validation + geometric hashing for convergence escape
    if res.success:
        initial_centers = np.column_stack([res.x[0::3], res.x[1::3]])
        initial_radii = res.x[2::3]
        # Add dynamic constraint reconfiguration with spatial hashing
        spatial_hashes = np.random.rand(n, 2) * 0.02  # Small perturbation for escape
        for i in range(n):
            res.x[3*i] += spatial_hashes[i, 0] * np.sqrt(initial_radii[i]) / 0.3
            res.x[3*i+1] += spatial_hashes[i, 1] * np.sqrt(initial_radii[i]) / 0.3
        # Re-run optimization with perturbed centers
        res = minimize(neg_sum_radii, res.x, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 900, "ftol": 1e-12})
    
    # Progressive radius expansion phase: optimize from largest to smallest, enforcing non-overlap
    # Use a hybrid approach: first global expansion, then local fine-tuning
    # We'll implement a two-phase expansion with validation after expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute initial max sum, but don't use it directly for expansion logic
        # Instead, compute expansion factors based on inter-distance ratios
        
        # Vectorized distance matrix using broadcasting for efficiency
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate allowed expansion: distance - sum of current radii
        # We can calculate expansion per index as (distance - r_i - r_j) / (r_i + r_j)
        # But need to handle edge cases properly
        
        # Generate a list of expansion factors
        expansion_factors = []
        for i in range(n):
            max_growth = 0
            for j in range(n):
                if i != j:
                    dist = dists[i, j]
                    min_radius = min(radii[i], radii[j])
                    if dist < (radii[i] + radii[j]) - 1e-8:
                        # Cannot expand - already overlapping
                        max_growth = 0
                        break
                    # Calculate potential expansion
                    if dist > (radii[i] + radii[j]):
                        # There's unused space, compute how much we can grow both
                        delta = dist - (radii[i] + radii[j])
                        expansion_i = delta * 0.5 / (radii[i] + np.finfo(float).eps)
                        expansion_j = delta * 0.5 / (radii[j] + np.finfo(float).eps)
                        # Take smaller of the two for more conservative growth
                        max_growth_candidate = min(expansion_i, expansion_j)
                        if max_growth_candidate > max_growth:
                            max_growth = max_growth_candidate
            expansion_factors.append(max_growth)
        
        # Weight expansion factors by distance from edges to prioritize inner circles
        distance_from_edges = 1 - np.max(np.abs(centers - [0.5, 0.5]), axis=1)
        weight = distance_from_edges * 0.8 + 0.2  # 80% weight to inner circles
        weighted_expansion = np.clip(expansion_factors * weight, 0, 0.005)  # Cap expansion at 0.5% per circle
        
        # Apply expansion with dynamic constraint validation
        # Need to make adjustments to avoid overlap, but we do it carefully
        # Initialize new_radii with current radii and apply expansion
        new_radii = radii.copy() + weighted_expansion[:n] * (1 + 0.05 * np.random.rand(n))
        
        # Try to optimize under new radii constraint
        # Create a version where only radii are expanded, and positions are fixed
        fixed_centers = centers.copy()
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Validate expanded constraints
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = (fixed_centers[i, 0] - fixed_centers[j, 0])
                dy = (fixed_centers[i, 1] - fixed_centers[j, 1])
                dist = np.sqrt(dx**2 + dy**2)
                if dist < new_radii[i] + new_radii[j] - 0.5e-6:  # 0.5e-6 to avoid floating-point errors
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # All constraints valid - proceed with new radii
            v = expanded_v
        else:
            # If overlap still exists, we need to perform a local radius optimization
            # This is a fallback and should not be necessary if the above logic worked
            # We'll run a final optimized step under the current centers to adjust radii
            temp_centers = centers.copy()
            temp_radii = radii.copy()
            temp_radii = temp_radii * (1 + (weighted_expansion[:n] * 0.8))
            temp_radii = np.clip(temp_radii, 1e-6, 0.5)
            
            # Re-optimized radii under the current centers to find maximum feasible sum
            # This is a fallback constraint for when the expansion caused overlap
            # Create a constraint-based optimization for radius only
            final_constr = [con for con in cons if con['type'] == 'ineq']
            
            # Define objective with the new centers
            def final_neg_sum_radii(radii_vector):
                # Keep the center positions fixed, only vary radii
                temp_v = v.copy()
                temp_v[2::3] = radii_vector
                return -np.sum(radii_vector)
            
            # Bounds for final optimization
            final_bounds = [(1e-6, 0.5) for _ in range(n)]
            
            # Run final radius only optimization with the current centers
            # Add constraints to prevent overlap in this phase
            final_cons = []
            for i in range(n):
                for j in range(i+1, n):
                    def final_overlap_func(v, i=i, j=j, centers=temp_centers):
                        dx = centers[i, 0] - centers[j, 0]
                        dy = centers[i, 1] - centers[j, 1]
                        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                    final_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: final_overlap_func(v, i, j, temp_centers)})
            
            final_res = minimize(final_neg_sum_radii, temp_radii, method="SLSQP", bounds=final_bounds, constraints=final_cons,
                                options={"maxiter": 800, "ftol": 1e-12})
            
            if final_res.success:
                v = final_res.x
            
    # Final cleanup: clip radii to be at least 1e-6 and not exceed 0.5
    v = np.clip(v, 0, 0.5)  # Clamp all values between 0 and 0.5
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]
    radii = np.clip(radii, 1e-6, None)  # Ensure radii are at least 1e-6
    
    # Final validation to ensure correctness
    # This is redundant due to constraint enforcement, but adds a safeguard
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12 or
            y - r < -1e-12 or y + r > 1 + 1e-12):
            # If any circle is outside, adjust its position to stay inside
            if x - r < -1e-12:
                v[3*i] = r
            if x + r > 1 + 1e-12:
                v[3*i] = 1 - r
            if y - r < -1e-12:
                v[3*i+1] = r
            if y + r > 1 + 1e-12:
                v[3*i+1] = 1 - r
            # Re-evaluate the final result
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
    
    return centers, radii, float(radii.sum())