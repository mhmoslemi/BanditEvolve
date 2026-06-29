import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    # Grid with flexible column and row configuration
    # Introduce dynamic staggered spacing for tighter packing
    # Initial grid layout: 5 columns, 6 rows with offset row spacing
    cols_initial = 5
    rows_initial = (n + cols_initial - 1) // cols_initial
    stagger = 0.2 / cols_initial  # Stagger for row-based displacement
    # Generate more diversified spatial positions
    xs = []
    ys = []
    for i in range(n):
        row = i // cols_initial
        col = i % cols_initial
        # Base x coordinate with column spacing
        x_center = (col + 0.5) / cols_initial
        # Base y coordinate with row spacing
        y_center = (row + 0.5) / rows_initial
        # Add dynamic jitter for breaking symmetry
        jitter_x = (np.random.rand() - 0.5) * 0.08
        jitter_y = (np.random.rand() - 0.5) * 0.08
        
        # Stagger alternate rows with controlled offset
        if row % 2 == 1:
            x_center += (stagger + np.random.rand() * 0.02) * np.sqrt(1.0 + (row % 3 == 0)/2)
        
        # Apply jitter and check for boundaries
        x = x_center + jitter_x
        y = y_center + jitter_y
        
        # Additional spatial perturbation for dynamic spatial constraints
        if i % 3 == 0 and not (row >= rows_initial - 2):
            x += (np.random.rand() - 0.5) * 0.03
            y += (np.random.rand() - 0.5) * 0.03
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation: adaptive based on spatial constraints and packing density
    # Use a more dynamic initial radius that considers column spacing
    avg_spacing_initial = (1.0 - 2*np.min([cols_initial, rows_initial]) * 0.02) / cols_initial
    r0_base = 0.32
    r0 = r0_base * (avg_spacing_initial / (1.0 - 2*r0_base)) * 0.98  # Account for radius expansion potential
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Define bounds with 3*n entries, strict lower bounds for radii
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.48)]  # Adjusted upper bound for radii
    
    # Objective function: -sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Optimizer constraints: boundary constraints (inequalities)
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
    
    # Overlap constraints (inequalities)
    # Use vectorized distance calculation to avoid repeated loops (vectorization via broadcasting)
    # Generate constraints in bulk with careful index handling
    for i in range(n):
        for j in range(i + 1, n):
            # Create constraint for i-j pair with closure
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                         - (v[3*i+2] + v[3*j+2])**2})
    
    # 1st optimization stage: fine-tuning with initial configuration
    # Increase max iterations and tighten tolerance to allow for more refined solution
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "eps": 1e-10})
    
    if res.success:
        # Post-optimization spatial reconfiguration using geometric hashing
        # Perturb positions with non-linear transformation based on radius and position
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute a spatial hash that depends on both radius and position
        # Scale hash by radius to give more flexibility to smaller circles
        spatial_hash = np.random.rand(n, 2) * 0.04  # 4% max perturbation
        hash_factor = radii / (radii.max() + 1e-6)  # Normalize spatial factors
        perturbed_v = v.copy()
        
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * hash_factor[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * hash_factor[i]
        
        # 2nd optimization: refine configuration around the perturbed space
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10})
    
    # 3rd stage: targeted expansion, with more sophisticated constraint-based expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances with broadcasting for efficient vectorization
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate a robust measure of least constrained point: max minimal distance to other circles
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Determine the maximum possible expansion based on current configuration
        # Use non-overlapping safety margin of 1e-12
        # Use vectorized checks to validate the expansion
        def is_config_valid(v, constraints, indices):
            # Get centers and radii
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < (radii[i] + radii[j]) - 1e-12:
                        return False
            return True
        
        # Define a vectorized expansion method
        def expand_radii_with_constraints(v, radii, expansion_factor, max_iterations=20):
            for _ in range(max_iterations):
                current_radii = v[2::3]
                new_radii = current_radii * expansion_factor
                # Clip radii to 0.48 to maintain feasibility
                new_radii = np.clip(new_radii, 1e-5, 0.48)
                # Apply new radii
                new_v = v.copy()
                new_v[2::3] = new_radii
                if is_config_valid(new_v, cons, (0, 1, 2, 3, 4, 5, 6, 7)):
                    # Keep the expanded version
                    v = new_v
                    radii = new_radii
                    break
            return v, radii
        
        # Determine expansion scale based on least constrained point and current sum
        # Try dynamic expansion, with adaptive scaling based on current configuration
        expansion_factor = 1.0
        for _ in range(3):
            expanded_v, expanded_radii = expand_radii_with_constraints(v, radii, expansion_factor)
            if is_config_valid(expanded_v, cons, (0, 1, 2, 3, 4, 5, 6, 7)):
                v = expanded_v
                radii = expanded_radii
                # Adjust expansion factor dynamically
                expansion_factor *= 1.002
            else:
                break
        
        # Final optimization: refine the expanded configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.48)  # Clamp to maximum feasible radius
    return centers, radii, float(radii.sum())