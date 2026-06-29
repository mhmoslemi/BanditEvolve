import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Random seed initialization to ensure consistent and reproducible starting points
    np.random.seed(42)
    
    # Initialize centers with spatial hashing and grid refinement
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset with tighter bounds for better spread and less clustering
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Staggered grid to avoid alignment
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation with improved scaling and adaptive spatial awareness
    base_radius = 0.33 / cols - 1e-3
    r0 = base_radius * np.ones(n)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # Bounds for optimization variables (length 3*n)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint generation with delayed lambda binding
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized pairwise distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with tightened tolerances and increased iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-12})

    # First reconfiguration: dynamic spatial hashing with directional expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric spatial hashing and asymmetric directional perturbation
        spatial_hash = np.random.rand(n, 2) * 0.03
        perturbed_v = v.copy()
        for i in range(n):
            # Directional perturbation with spatial awareness
            direction = np.random.choice(['x', 'y'])
            magnitude = np.random.uniform(0.001, 0.004)
            if direction == 'x':
                perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii) * 0.8)
            else:
                perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii) * 0.8)
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute dense pairwise distances and identify isolated circle
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Maximize minimum distances for isolation
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate total possible expansion based on density and spatial distribution
        current_total = np.sum(radii)
        target_total_sum = current_total + 0.008  # 0.008 is 0.3% of the optimal expected range
        expansion_factor = (target_total_sum - current_total) / (n - 1) * (current_total / np.sum(radii))
        
        # Create expanded radii vector with spatial intelligence
        new_radii = radii.copy()
        # Targeted expansion for least constrained circle with adaptive over-expansion
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # 20% over-target for exploration
        # Stochastic expansion for all other circles with directional variance
        for i in range(n):
            if i != least_constrained_idx:
                direction = np.random.choice(['x', 'y'])
                # Directional perturbation to avoid gridlock
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())
                # Add directional perturbation for spatial dispersion
                if direction == 'x':
                    perturbation = np.random.uniform(-0.03, 0.03) * (radii[i] / np.mean(radii))
                    new_radii[i] += expansion_i * (1 + perturbation)
                else:
                    perturbation = np.random.uniform(-0.03, 0.03) * (radii[i] / np.mean(radii))
                    new_radii[i] += expansion_i * (1 + perturbation)
        
        # Apply expansion with iterative validation
        iterations = 0
        while iterations < 5:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate all pairwise distances with precision
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Linear backtracking if invalid configuration
                new_radii = radii + (new_radii - radii) * (0.95 + 0.03 * iterations)
                iterations += 1
        
        # Update vector with optimized new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final reoptimization with high precision and limited iteration budget
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 700, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())