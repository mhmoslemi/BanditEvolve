import numpy as np

def run_packing():
    n = 26
    cols = 5  # 5 columns for 25 circles, 1 extra row for the 26th
    rows = (n + cols - 1) // cols
    
    # Initialize positions with optimized staggered grid and random perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        
        # Add subtle random perturbations that maintain grid structure
        # Avoid large jumps to preserve initial cluster integrity
        x_perturb = np.random.uniform(-0.035, 0.035)
        y_perturb = np.random.uniform(-0.035, 0.035)
        
        # Ensure no overlap with grid edges
        x_center = x_base + x_perturb
        y_center = y_base + y_perturb
        
        # Apply row-based stagger
        if row % 2 == 1:
            x_center += 0.35 / cols  # Stagger by half the column spacing
        
        # Clamp x_center to avoid boundary issues, preserving perturbation
        x_center = np.clip(x_center, 1e-6, 1.0 - 1e-6)
        y_center = np.clip(y_center, 1e-6, 1.0 - 1e-6)
        
        xs.append(x_center)
        ys.append(y_center)
    
    r0 = 0.33 / cols - 1e-4  # Slightly smaller initial radii for better convergence
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries for 26 circles

    # Vectorized constraint evaluator with numpy broadcasting
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    ###### CONSTRAINT SYSTEM DESIGN (CRITICAL PATH)######
    constraints = []
    
    # Boundary constraints using numpy broadcasting
    for i in range(n):
        # Left boundary: x - r >= 0
        # Right boundary: x + r <= 1
        # Bottom boundary: y - r >= 0
        # Top boundary: y + r <= 1
        constraints.append(
            {"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]}  # Right
        )
        constraints.append(
            {"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]}      # Left
        )
        constraints.append(
            {"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]}  # Top
        )
        constraints.append(
            {"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]}      # Bottom
        )

    # Distance constraints using numpy broadcasting
    # Create a vectorized constraint matrix to optimize performance
    dist_constraints = []
    
    for i in range(n):
        for j in range(i+1, n):
            idx_i = 3*i
            idx_j = 3*j
            def distance_constr(v, i=i, j=j):
                dx = v[idx_i] - v[idx_j]
                dy = v[idx_i+1] - v[idx_j+1]
                return dx*dx + dy*dy - (v[idx_i+2] + v[idx_j+2])**2
            dist_constraints.append({"type": "ineq", "fun": distance_constr})
    
    # Append distance constraints to main constraint list
    constraints.extend(dist_constraints)

    ###### OPTIMIZATION PHASES: MULTI-STEP TACTIC IMPLEMENTATION ###### 
    # Phase 1: Initial optimization with tighter gradient tolerance
    res = minimize(
        neg_sum_radii, v0, 
        method='SLSQP', 
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9, "eps": 1e-8, "disp": False}
    )
    
    # Phase 2: Perturbation-based reconfiguration of key circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Identify dynamic interacting pairs based on proximity
        # Use vectorized operations instead of nested loops
        dx_full = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy_full = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx_full**2 + dy_full**2)
        interactions = np.sum(dists, axis=1)
        
        # Select the most dynamically interacting pair (top 2)
        top_indices = np.argsort(interactions)[-2:]
        i, j = top_indices
        
        # Perturb positions of interacting pair with a soft constraint
        perturbation = 0.035  # Small scale modification to enable reordering
        v[3*i] += np.random.uniform(-perturbation, perturbation)
        v[3*i+1] += np.random.uniform(-perturbation, perturbation)
        v[3*j] += np.random.uniform(-perturbation, perturbation)
        v[3*j+1] += np.random.uniform(-perturbation, perturbation)
        
        # Reoptimize with perturbed configuration
        res = minimize(
            neg_sum_radii, v, 
            method='SLSQP', 
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 400, "ftol": 1e-9, "eps": 1e-8, "disp": False}
        )
    
    # Phase 3: Targeted radius expansion on least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distance matrix with numpy broadcasting
        dx_full = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy_full = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        distances = np.sqrt(dx_full**2 + dy_full**2)
        
        # Find the circle with the most isolation (minimum proximity)
        min_proximity = np.min(distances, axis=1)
        isolated_idx = np.argmin(min_proximity)  # Circle with least interaction
        
        # Check if it's a realistic isolation (at least one other circle within 1.5x radius)
        min_distance_to_others = np.min(distances[isolated_idx, :])
        if min_distance_to_others < radii[isolated_idx] * 1.5:
            isolated_idx = np.argmax(min_proximity)  # Fallback to the most isolated
        
        # Calculate expansion potential based on total sum
        total_sum = np.sum(radii)
        expansion_limit = 0.0065  # Targeted expansion of 0.0065 total
        expansion_per = expansion_limit / (n - 1) * (total_sum / np.sum(radii)) * 1.1  # 10% over-expansion strategy
        
        # Create expansion vector with targeted expansion
        new_radii = radii.copy()
        new_radii[isolated_idx] += expansion_per * 1.2  # 20% extra for dynamic expansion
        
        # Apply expansion while maintaining constraints through iteration
        while True:
            # Create temporary configuration with expanded radii
            temp_v = v.copy()
            temp_v[2::3] = new_radii
            
            # Check validity
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = temp_v[3*i] - temp_v[3*j]
                    dy = temp_v[3*i+1] - temp_v[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, back off expansion by 20% and retry
                new_radii = radii + (new_radii - radii) * 0.85
        
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Reoptimize with expanded configuration
        res = minimize(
            neg_sum_radii, v_new, 
            method='SLSQP', 
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 400, "ftol": 1e-9, "eps": 1e-8, "disp": False}
        )
    
    # Final adjustment using geometric hashing for stability
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial perturbation based on relative sizes
        hash_map = np.random.rand(n, 2) * 0.025
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_map[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += hash_map[i, 1] * (radii[i] / np.mean(radii))
        
        # Reoptimize with perturbed configuration
        res = minimize(
            neg_sum_radii, perturbed_v, 
            method='SLSQP', 
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 400, "ftol": 1e-9, "eps": 1e-8, "disp": False}
        )
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())