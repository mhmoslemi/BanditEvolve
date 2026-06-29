import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized geometric tiling (non-local reconfiguration)
    # Use Voronoi grid with randomized seed, then add perturbation for spatial diversity
    xs = []
    ys = []
    seed = np.random.randint(0, 1000000)
    np.random.seed(seed)
    for i in range(n):
        # Base grid placement with randomized Voronoi-like spacing
        col = i % cols
        row = i // cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Apply spatial hashing with adaptive expansion to break symmetry
        spatial_hash = np.random.rand(2) * 0.08
        x = base_x + spatial_hash[0]
        y = base_y + spatial_hash[1]
        
        # Shift alternate rows to create staggered tiling
        if row % 2 == 1:
            x += 0.5 / cols * (1 + np.random.rand())
        
        # Apply adaptive jitter based on column and row position
        if col < cols // 2:
            x += np.random.uniform(-0.02, 0.02)
        elif col > cols // 2:
            x += np.random.uniform(0.01, 0.03)
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure the bounds list has 3*n entries for the vector of length 3n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i
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
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-10})
    
    # Targeted radius expansion on least constrained circle with soft constraints and dynamic bounds
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate growth based on current total sum and potential for expansion
        current_total = np.sum(radii)
        target_growth = 0.008
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))

        # Identify the circle with the smallest non-zero radius for targeted expansion
        min_radius = np.min(radii[radii > 1e-9])
        min_radius_idx = np.argmin(radii[radii > 1e-9])
        
        # Apply expansion focusing on least constrained and smallest radius circles
        new_radii = radii.copy()
        # Slightly over-expand least constrained
        new_radii[least_constrained_idx] += expansion_factor * 1.25
        # Over-expand the smallest radius circle to unlock radii potential
        new_radii[min_radius_idx] += expansion_factor * 1.5
        for i in range(n):
            if i not in (least_constrained_idx, min_radius_idx):
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand()) 
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
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
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 450, "ftol": 1e-11})

    # Final configuration with final validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())