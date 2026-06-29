import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Optimized initialization with enhanced initial spread and geometric awareness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Spatial-aware offset with controlled randomness and staggered adjustment
        x = x_center + np.random.uniform(-0.04, 0.04) + (row % 2) * 0.02 / cols
        y = y_center + np.random.uniform(-0.04, 0.04) + (row % 2) * 0.02 / rows
        xs.append(x)
        ys.append(y)
    
    # Adaptive radius calculation based on density and spatial distribution
    grid_density = (1.0 - (cols - 1) / cols) * (1.0 - (rows - 1) / rows)
    base_radius = (0.32 + grid_density * 0.04) / cols - 1e-3
    r0 = base_radius * np.ones(n)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensuring 3n entries for the decision vector
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Boundary constraint generation with vectorized and lambda-preserved capture
    cons = []
    for i in range(n):
        # Left edge + radius <= 1 constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right edge - radius >= 0 constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom edge + radius <= 1 constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top edge - radius >= 0 constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Overlap constraint generation with lambda-captured indices and vectorization
    for i in range(n):
        for j in range(i + 1, n):
            # Distance squared between i and j circles - (radii_i + radii_j)^2
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})
    
    # Initial optimization with tighter tolerance and iterative refinement
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Asymmetric spatial hashing for targeted perturbation with dynamic scaling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create spatial hashing vector with adaptive scaling based on radii
        spatial_hash = np.random.rand(n, 2) * 0.03
        perturbed_v = v.copy()
        for i in range(n):
            scaling = (radii[i] / np.mean(radii)) * 1.2
            perturbed_v[3*i] += spatial_hash[i, 0] * scaling
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scaling
        
        # Perform spatial reconfiguration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # Enhanced targeted expansion of least constrained circle with adaptive growth
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix using broadcasting for all pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle: max of minimum distances
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate growth from current total to potential expansion
        current_total = np.sum(radii)
        target_growth = 0.0075  # Target growth fraction added to total
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create new radii with expansion on least constrained circle and adaptive
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.15  # 15% over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.08 * np.random.rand())  # 8% stochastic expansion
        
        # Validate and refine expansion
        iterations = 0
        while iterations < 3:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate pairwise distances
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
                # Adjust by reducing growth factor
                new_radii = radii + (new_radii - radii) * 0.95
                iterations += 1
        
        # Update vector with refined expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization to lock in expanded radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())