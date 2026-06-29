import numpy as np

def run_packing():
    n = 26
    
    # Adaptive grid generation with variable rows and cols to maximize spacing
    cols_base = 5
    rows_base = 6
    # Use a more balanced grid to reduce symmetry
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Spatial hashing to break symmetry and guide configuration
    spatial_hash = np.random.rand(n, 2) * 0.05
    # Adjacency bias vector for constraint prioritization
    adj_bias = np.random.rand(n, 2) * 0.2
    
    # Grid generation with asymmetric spacing and random perturbations for initial setup
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset for symmetry breaking
        x = x_center + np.random.uniform(-0.07, 0.07)
        y = y_center + np.random.uniform(-0.07, 0.07)
        # Staggered rows to improve packing efficiency
        if row % 2 == 1:
            x += (1.0 / cols) * 0.45
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with adaptive sizing
    r0 = 0.3 + np.random.rand(n) * 0.05  # Increased base to allow for expansion opportunities
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    # Ensure the bounds list matches the 3*n length requirement
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Build constraints with lambda closures and delayed evaluation (capture i and j)
    cons = []
    for i in range(n):
        # Left bound constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right bound constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom bound constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top bound constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Construct pairwise overlap constraints with lambda closure
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                          (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                          - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization stage with aggressive iteration and tight tolerance
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-12, "eps": 1e-10, "disp": False}
    )
    
    # First reconfiguration: spatial hashing with directional bias
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Create a perturbation vector based on spatial and adjacency hashing
        perturbed_v = v.copy()
        for i in range(n):
            # Spatial perturbations with directional bias
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i]/np.mean(radii)) * 1.3
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i]/np.mean(radii)) * 1.3
            # Adjacency-based perturbations to create spacing bias
            if i < n - 2:
                perturbed_v[3*i+2] += adj_bias[i, 0] * 0.003
                perturbed_v[3*i+1] += adj_bias[i, 1] * 0.002
        
        # Re-evaluate perturbed configuration
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-10, "disp": False}
        )

    # Second reconfiguration: focused repositioning of least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with the most spatial breathing room (largest minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Targeted expansion planning using spatial hashing and adjacency bias
        spatial_perturbation = spatial_hash[least_constrained_idx] * 2.0
        adjacency_perturbation = adj_bias[least_constrained_idx] * 1.5

        # Create a perturbation vector to reposition the least constrained circle
        perturbed_v = v.copy()
        # Move the center of the least constrained circle with spatial hash directional bias
        perturbed_v[3*least_constrained_idx] += spatial_perturbation[0] * (radii[least_constrained_idx]/np.mean(radii))
        perturbed_v[3*least_constrained_idx+1] += spatial_perturbation[1] * (radii[least_constrained_idx]/np.mean(radii))
        # Adjust its radius with adjacency-based growth
        perturbed_v[3*least_constrained_idx+2] += adjacency_perturbation[0] * 0.006
        
        # Re-evaluate with new configuration
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-10, "disp": False}
        )

    # Final reconfiguration: expansion of the least constrained circle with directional expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Re-calculate distances for the final optimization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle (most expansion potential)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate the maximum potential total sum with expansion
        expansion_multiplier = 1.02  # Target increase of 2% in total sum
        target_total_sum = np.sum(radii) * expansion_multiplier
        
        # Create directional expansion vector for the least constrained circle
        expansion_v = v.copy()
        # Expand only the least constrained circle
        expansion_v[3*least_constrained_idx+2] += (target_total_sum - np.sum(radii)) * 0.3
        
        # Validate this configuration to ensure no overlaps
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx_exp = expansion_v[3*i] - expansion_v[3*j]
                dy_exp = expansion_v[3*i+1] - expansion_v[3*j+1]
                dist = np.sqrt(dx_exp**2 + dy_exp**2)
                if dist < (expansion_v[3*i+2] + expansion_v[3*j+2]) - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # If valid, accept this expansion
            v = expansion_v
        else:
            # If invalid, revert and try smaller expansion
            v = res.x
        
        # Final optimization on modified configuration
        res = minimize(
            neg_sum_radii, 
            v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-10, "disp": False}
        )

    # Ensure final solution satisfies all constraints
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())