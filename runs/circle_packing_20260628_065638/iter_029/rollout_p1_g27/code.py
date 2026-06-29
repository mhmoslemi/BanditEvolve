import numpy as np
import warnings

def run_packing():
    n = 26
    cols, rows = 6, 5  # Dynamic grid that allows flexible staggered configurations
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid: offset in x for even rows, shifted for odd rows (hexagonal stagger)
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add small randomized asymmetric shift for initial spread
        shift_x = np.random.uniform(-0.05, 0.05)
        shift_y = np.random.uniform(-0.05, 0.05)
        
        # Staggering for hexagonal layout (shift alternate rows)
        if row % 2 == 1:
            x_center += 0.5 / cols
        x = x_center + shift_x
        y = y_center + shift_y
        
        # Boundary check for initial positions
        if x < 0 or y < 0 or x > 1 or y > 1:
            # Re-center if position is out of bound due to shifts
            x = max(0, min(1, x))
            y = max(0, min(1, y))
        
        xs.append(x)
        ys.append(y)
    
    # Calculate base radii based on square packing efficiency with adaptive adjustment
    # In square grid, optimal radius is about (sqrt(2)/4) * 1 / cols / sqrt(rows) 
    # Here we refine for hexagonal layout by adjusting for rows
    r0_base = (1.0 / (cols * np.sqrt(3))) * 1.05  # Slightly higher to allow expansion
    # Adjust radii based on row spacing to improve vertical packing
    r0 = np.full(n, r0_base) * (1.0 + np.random.uniform(-0.015, 0.015))
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Define bounds with strict length enforcement to match 3n-length vector
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensuring 3n elements

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize radii by minimizing negative sum

    # Vectorized constraints for boundary checks, using closure capture with i
    cons = []
    for i in range(n):
        # Left: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right: 1.0 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top: 1.0 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})

    # Vectorized constraints for non-overlapping, using i and j closures
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # First initialization pass with tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-12, 
                                             "eps": 1e-10, "disp": False})

    # Reconfiguration loop to apply enhanced spatial dissection
    best_sum = 0
    best_config = (np.zeros((n, 2)), np.zeros(n), 0.0)

    for _ in range(3):  # 3 iterations to apply dissection
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])

            # Compute spatial graph for adjacency analysis
            dists = np.zeros((n, n))
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)
            adj_matrix = (dists < 0.3)  # Threshold for local adjacency in sparse packing
            
            # Identify the two most dynamically interacting (adjacent) circles
            # Compute adjacency degree for every circle
            adj_degrees = np.sum(adj_matrix, axis=1)
            most_interacting = np.argsort(adj_degrees)[-2:]  # Top 2 most interacting

            # Perform dissection on two circles to reconfigure their relationship
            # Create a new perturbed configuration with dissection
            spatial_hash = np.random.rand(n, 2) * 0.06
            directional_hash = np.random.rand(n, 2) * 0.04
            perturbed_v = v.copy()
            for i in range(n):
                # Move based on spatial hashing and adjacency
                perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) 
                perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
                if i != most_interacting[0] and i != most_interacting[1]:
                    # For non-interacting circles: small directional bias
                    perturbed_v[3*i] += directional_hash[i, 0] * 0.002 * (1.0 + 0.1 * np.random.rand())
                    perturbed_v[3*i+1] += directional_hash[i, 1] * 0.001 * (1.0 + 0.1 * np.random.rand())
            
            # Dissect the two interacting circles by applying a controlled perturbation
            # Move the first interacting circle
            perturbed_v[3*most_interacting[0]] += np.random.uniform(-0.02, 0.02)
            perturbed_v[3*most_interacting[0]+1] += np.random.uniform(-0.02, 0.02)
            # Move the second interacting circle inversely
            perturbed_v[3*most_interacting[1]] -= np.random.uniform(-0.02, 0.02)
            perturbed_v[3*most_interacting[1]+1] -= np.random.uniform(-0.02, 0.02)

            # Second optimization with reconfigured centers and perturbed radii
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11,
                                                     "eps": 1e-10, "disp": False})

        if res.success:
            # Calculate performance metric to track improvement
            sum_r = float(np.sum(res.x[2::3]))
            if sum_r > best_sum:
                best_sum = sum_r
                best_config = (
                    np.column_stack([res.x[0::3], res.x[1::3]]),
                    res.x[2::3], 
                    sum_r
                )
        else:
            # Fallback to previous best if optimization fails
            pass  # Fall back to prior state

    # Apply a controlled expansion to the least constrained circle with adjacency awareness
    final_centers = best_config[0]
    final_radii = best_config[1]
    if best_config[0].size > 0 and best_config[1].size > 0:
        dists = np.zeros((n, n))
        dx = final_centers[:, np.newaxis, 0] - final_centers[np.newaxis, :, 0]
        dy = final_centers[:, np.newaxis, 1] - final_centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Expand it with directional bias to adjacent circles based on adjacency graph
        expansion_factor = 0.008  # Growth based on current sum and target improvement
        current_total = np.sum(final_radii)
        expansion_scale = expansion_factor * (current_total / np.mean(final_radii))  # Adaptive scaling
        
        # Initial expansion to least constrained circle with directional boost
        new_radii = final_radii.copy()
        new_radii[least_constrained_idx] += expansion_scale * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                adj_weight = np.linalg.norm(final_centers[least_constrained_idx] - final_centers[i])
                if adj_weight < 0.3:  # Boost expansion for nearby circles
                    expansion = expansion_scale * (1.1 + 0.3 * np.random.rand())
                else:
                    expansion = expansion_scale * (1.0 + 0.2 * np.random.rand())
                new_radii[i] += expansion
        
        # Apply expansion iteratively with constraint checking
        valid_expansion = False
        while True:
            expanded_v = best_config[0].copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                valid_expansion = True
                break
            else:
                # Decrease expansion if invalid
                new_radii = final_radii + (new_radii - final_radii) * 0.98
        
        if valid_expansion:
            # Re-run optimization with expanded radii and new configuration
            expanded_v = best_config[0].copy()
            expanded_v[2::3] = new_radii
            res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11,
                                                     "eps": 1e-10, "disp": False})
        
        if not valid_expansion or not res.success:
            # Fallback to best_config if expansion fails
            return best_config[0], best_config[1], best_config[2]

    # Final fallback to the best configuration found during the process
    return best_config[0], best_config[1], best_config[2]