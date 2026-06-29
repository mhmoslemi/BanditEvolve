import numpy as np

def run_packing():
    n = 26
    cols = 6  # Use more columns to spread circles and reduce clustering
    rows = (n + cols - 1) // cols
    
    # Define initial position clusters with adaptive jitter and spatial dispersion
    xs = []
    ys = []
    
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base positions with dynamic resolution
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Adaptive jitter based on row proximity to prevent clustering at edges
        row_jitter = np.random.uniform(-0.04, 0.04) if row < 2 or row > rows - 3 else np.random.uniform(0, 0)
        col_jitter = np.random.uniform(-0.03, 0.03) if col < 1 or col > cols - 2 else np.random.uniform(0, 0)
        
        # Apply staggered grid adjustment
        if row % 2 == 1:
            x_center += 0.5 / cols * 1.15  # Increased offset for better staggered distribution
        
        x = x_center + col_jitter
        y = y_center + row_jitter
        
        # Apply spatial perturbation for more randomized distribution
        x += np.random.uniform(-0.02, 0.02)
        y += np.random.uniform(-0.02, 0.02)
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.5 / cols - 1e-2  # Increase initial radius size for better expansion potential
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure all constraints match decision vector

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective function with gradient approximation

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-8, "eps": 1e-8})
    
    # Spatial reconfiguration via random geometric hashing with adaptive scaling
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Generate spatial hash with scale based on spatial density
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * np.clip(radii[i]/np.mean(radii), 0.8, 1.2)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * np.clip(radii[i]/np.mean(radii), 0.8, 1.2)
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-9, "eps": 1e-9})

    # Targeted radius expansion on least constrained circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Distance matrix via broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion target with spatial density and radius normalization
        avg_distance = np.median(min_dists)
        avg_radius = np.mean(radii)
        expansion_factor = (avg_distance - avg_radius * 1.12) * 0.75  # Dynamic expansion based on spacing
        
        # Apply radius expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Amplify expansion on key circle
        new_radii = np.clip(new_radii, 1e-6, 0.5)  # Apply hard limit per constraint
        
        # Validate expansion configuration
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
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
                # Decay expansion based on distance violation
                if np.any(new_radii < 1e-6):
                    new_radii = np.clip(new_radii, 1e-8, 0.5)
                else:
                    new_radii = radii + (new_radii - radii) * 0.97
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())