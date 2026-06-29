import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with geometric hashing, adaptive perturbation, and sparse clustering
    xs = []
    ys = []
    for i in range(n):
        base_row = i // cols
        base_col = i % cols
        x_center = (base_col + 0.5) / cols
        y_center = (base_row + 0.5) / rows
        
        # Adaptive geometric clustering: scale based on row and column distance
        row_diff = base_row - rows / 2
        col_diff = base_col - cols / 2
        dist = np.sqrt(row_diff**2 + col_diff**2)
        cluster_scale = 1.0 / (1.0 + 0.2 * dist)
        
        # Apply randomized geometric hashing for positional perturbation
        x = x_center + (np.random.rand() - 0.5) * 0.12 * cluster_scale
        y = y_center + (np.random.rand() - 0.5) * 0.12 * cluster_scale
        
        # Alternate row stagger for non-square grid
        if base_row % 2 == 1:
            x += 0.5 / cols * cluster_scale
        
        xs.append(x)
        ys.append(y)
    
    # Initial radii as relative to clustering scale and row/column indices
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n elements for 3*26

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with efficient closure capture
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
    
    # Vectorized overlap constraints with closure capture
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq", 
                "fun": (lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                        - (v[3*i+2] + v[3*j+2])**2)
            })

    # First optimization with enhanced convergence
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "eps": 1e-10})

    # Geometric reconfiguration phase with controlled perturbations
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling based on row/column distance
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            # Adaptive scaling based on cluster scale
            row = i // cols
            col = i % cols
            row_diff = row - rows / 2
            col_diff = col - cols / 2
            dist = np.sqrt(row_diff**2 + col_diff**2)
            cluster_scale = 1.0 / (1.0 + 0.2 * dist)
            
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * cluster_scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * cluster_scale
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Adaptive radius expansion targeting geometrically isolated nodes
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate geometric isolation metric: sum of inverse distances
        inverse_dists = np.where(dists > 1e-8, 1 / dists, 0)
        isolation = np.sum(inverse_dists, axis=1)
        
        # Find circle with minimal relative isolation (top-down optimization)
        least_isolated_idx = np.argmin(isolation)
        min_isolation = isolation[least_isolated_idx]
        
        # Calculate radius expansion factor based on isolation and current total
        current_total = np.sum(radii)
        target_growth = 0.006
        expansion_factor = target_growth * (current_total / (n - 1)) * (1 + 1.0 / min_isolation)
        
        # Create expansion vector with targeted expansion on least-isolated
        new_radii = radii.copy()
        new_radii[least_isolated_idx] += expansion_factor * 1.1
        for i in range(n):
            if i != least_isolated_idx:
                new_radii[i] += expansion_factor * np.random.uniform(0.8, 1.1)
        
        # Validate and refine expansion with local iteration
        iterations = 0
        while iterations < 3:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration using vectorized checking
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion and try again
                new_radii = radii + (new_radii - radii) * 0.92
                iterations += 1
        
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with refined configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())