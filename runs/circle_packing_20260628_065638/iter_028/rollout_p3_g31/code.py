import numpy as np

def run_packing():
    n = 26
    cols = 6  # 6x5 grid provides more flexibility
    rows = (n + cols - 1) // cols
    
    # Generate spatial grid with Gaussian-based stochastic perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Apply Gaussian distribution for spatial perturbation
        x_add = np.random.normal(loc=0, scale=0.06, size=1)
        y_add = np.random.normal(loc=0, scale=0.06, size=1)
        x = x_center + x_add[0]
        y = y_center + y_add[0]
        
        # Apply column-specific offset for better spacing
        if col <= 2:
            x -= 0.02  # Push left-side columns to left
        elif col >= 3:
            x += 0.02  # Push middle-right columns to right
        if row % 3 == 1:
            y += 0.02  # Push alternate rows down
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius guess based on grid size and spacing
    r0 = (0.3 / cols) - 1e-4
    
    # Construct decision vector
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds list of length 3*n for all circle parameters
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * n
    
    # Define negative sum objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Create constraints with lambda captures for correct indexing
    cons = []
    
    # Add boundary constraints for all circles
    for i in range(n):
        x_index = 3 * i
        y_index = 3 * i + 1
        r_index = 3 * i + 2
        
        # Left margin constraint: x - r >= 0
        cons.append({
            "type": "ineq",
            "fun": lambda v, i=i: v[3*i] - v[3*i+2]
        })
        # Right margin constraint: x + r <= 1
        cons.append({
            "type": "ineq",
            "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]
        })
        # Bottom margin constraint: y - r >= 0
        cons.append({
            "type": "ineq",
            "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]
        })
        # Top margin constraint: y + r <= 1
        cons.append({
            "type": "ineq",
            "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]
        })
    
    # Add overlap constraints for all circle pairs
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            })
    
    # First phase: initial optimization with spatial configuration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1500, "ftol": 1e-10})
    
    # If optimization was successful, perform asymmetric reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash for reconfiguration perturbation
        spatial_hash = np.random.randn(n, 2) * (radii / np.mean(radii)) * 0.08
        perturbed_v = v.copy()
        
        # Apply spatial hash only to the centers, not radii
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Perform spatial reconfiguration with optimized constraints
        # This phase optimizes only spatial positions to allow new spacing
        temp_res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds, constraints=cons,
                            options={"maxiter": 400, "ftol": 1e-11})
        
        # If spatial reconfiguration was successful, proceed
        v = temp_res.x if temp_res.success else v
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Targeted radius expansion on least constrained circle
        # Step 1: Compute minimum inter-circle distance for each circle
        dists = np.zeros((n, n))
        
        # Use broadcasting to compute all pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute minimum distance from each circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)  # Circle with the largest minimum distance
        
        # Step 2: Compute current radius sum
        current_total = np.sum(radii)
        # Apply controlled expansion to the least constrained circle
        expansion = 0.005  # 0.5% expansion of current total
        for i in range(n):
            if i == least_constrained_idx:
                proposed_radius = radii[i] + expansion
            else:
                proposed_radius = radii[i]
            
            # Check if the expanded configuration is feasible
            valid = True
            for j in range(n):
                if i != j:
                    dx_ij = centers[i, 0] - centers[j, 0]
                    dy_ij = centers[i, 1] - centers[j, 1]
                    dist_ij = np.sqrt(dx_ij**2 + dy_ij**2)
                    if dist_ij < proposed_radius + radii[j] - 1e-8:
                        valid = False
                        break
            if not valid:
                # Reduce expansion if needed
                expansion *= 0.95
        
        # Apply the expansion to the least constrained circle
        v[3*least_constrained_idx + 2] = radii[least_constrained_idx] + expansion
        
        # Step 3: Final optimization to stabilize the configuration
        final_res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds, constraints=cons,
                            options={"maxiter": 400, "ftol": 1e-12})
        
        # Final output
        v = final_res.x if final_res.success else v
        radii = np.clip(v[2::3], 1e-6, None)
        centers = np.column_stack([v[0::3], v[1::3]])
        return centers, radii, float(radii.sum())
    
    # If initial optimization failed, return default configuration
    v = v0
    radii = np.clip(v[2::3], 1e-6, None)
    centers = np.column_stack([v[0::3], v[1::3]])
    return centers, radii, float(radii.sum())