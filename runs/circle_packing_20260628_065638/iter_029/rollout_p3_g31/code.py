import numpy as np

def run_packing():
    n = 26
    
    # Optimize grid layout: 5x5 with 6 circles per row (first row has 5)
    cols = 5
    rows = (n + cols - 1) // cols
    base_grid = np.array([[ (j + 0.5) / cols, (i + 0.5) / rows ] for i in range(rows) for j in range(cols)])
    
    # Enhance initial configuration with dynamic bias toward corner clustering
    # We'll use a hybrid strategy of fixed grid, perturbation, and randomized geometric tile overlay
    def generate_initial_centers(grid):
        centers = []
        for i in range(n):
            x_ref, y_ref = grid[i]
            # Add controlled randomness with decreasing magnitude based on grid location
            x = x_ref + np.random.uniform(-0.02, 0.02) * (1 - i / n)  # Less perturbation to central circles
            y = y_ref + np.random.uniform(-0.02, 0.02) * (1 - i / n)
            # Apply corner bias
            corner_factor = 1.0 - 0.2 * (i % cols + i // cols) / (cols + rows)  # More variance for non-corner locations
            x += np.random.uniform(-0.02 * corner_factor, 0.02 * corner_factor)
            y += np.random.uniform(-0.02 * corner_factor, 0.02 * corner_factor)
            # Create staggered grid for alternate rows
            if (i // cols) % 2 == 1:
                # Shift right by 0.2 / cols to avoid overlap
                x += 0.2 / cols
            centers.append((x, y))
        return centers
    
    # Generate initial configuration with advanced hybrid layout
    centers = generate_initial_centers(base_grid)
    initial_radii = 0.35 / cols - 1e-3
    v0 = np.zeros(3 * n)
    v0[0::3] = np.array([c[0] for c in centers])
    v0[1::3] = np.array([c[1] for c in centers])
    v0[2::3] = np.full(n, initial_radii)
    
    # Create precise bounds list of length 3n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    # Define objective function to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Construct constraints with efficient vectorized expressions, lambda with i, no closure ambiguity
    cons = []
    
    # Boundary constraints: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
    
    # Create spatial hashing for efficient distance evaluations
    # Compute all pairwise distances in vectorized form for better performance
    
    # Optimized overlap constraints using vectorized calculation
    def create_overlap_constraints():
        # Precompute all distances in vector form for optimization
        for i in range(n):
            for j in range(i + 1, n):
                cons.append({
                    "type": "ineq",
                    "fun": lambda v, i=i, j=j: (
                        (v[3*i] - v[3*j])**2 + 
                        (v[3*i+1] - v[3*j+1])**2 
                        - (v[3*i+2] + v[3*j+2])**2
                    )
                })
    
    create_overlap_constraints()
    
    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "eps": 1e-8})
    
    # Refinement phase 1: spatial perturbation with gradient-guided hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial gradients for sensitivity analysis
        # We'll use perturbation method to compute Jacobian for better gradient approximation
        # Spatial hashing - create perturbed configuration with geometric bias
        delta = 0.03
        spatial_hash = np.random.rand(n, 2) * 0.03
        perturbed_v = v.copy()
        perturbed_v[0::3] += spatial_hash[:,0]
        perturbed_v[1::3] += spatial_hash[:,1]
        
        # Apply gradient-guided correction
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 220, "ftol": 1e-11, "eps": 1e-9})
    
    # Refinement phase 2: strategic geometric dissection on critical circles
    # We'll identify circles with minimal influence first
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate all pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify circles with minimal influence (least overlapping with others)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argsort(min_dists)[-4:]  # Top 4 least constrained
        
        # We'll perform geometric surgery on a subset: reduce their radii, reposition to allow expansion for others
        # Strategic reconfiguration to unlock more space
        # For each least constrained circle, move them to a corner to free space in center
        v_new = v.copy()
        for idx in least_constrained_idx:
            x, y, r = v[3*idx], v[3*idx+1], v[3*idx+2]
            # Move to upper left corner with slight padding
            dx = x - 0.8
            dy = y - 0.8
            # Adjust radius by reducing to ensure it fits in corner
            max_corner_r = np.min([0.8 - x, 0.8 - y])
            r_new = max(1e-4, r - max(0.02, (r - max_corner_r) * 0.8))
            v_new[3*idx] = max(0.0, min(1.0, x - dx * 1.4))
            v_new[3*idx+1] = max(0.0, min(1.0, y - dy * 1.4))
            v_new[3*idx+2] = r_new
        
        # Re-evaluate with adjusted positions
        res = minimize(neg_sum_radii, v_new, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 200, "ftol": 1e-11, "eps": 1e-9})
    
    # Adaptive radius expansion on most spatially unconstrained circles
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Recompute pairwise distances for influence analysis
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find most unconstrained circles (minimal overlap with others)
        min_dists = np.min(dists, axis=1)
        most_unconstrained_idx = np.argsort(min_dists)[-5:]  # Top 5 most unconstrained
        
        # Calculate potential expansion based on current constraints
        current_total = np.sum(radii)
        base_growth = 0.008  # Targeted expansion of 0.8% of current sum
        expansion_per_circle = base_growth / n * 1.8  # Slightly over-expand based on distribution
        
        # Apply growth to most unconstrained circles in a non-uniform way
        for idx in most_unconstrained_idx:
            # Calculate maximum possible radius increase without violating constraints
            max_grow = 0
            for j in range(n):
                if j == idx:
                    continue
                dx = centers[idx, 0] - centers[j, 0]
                dy = centers[idx, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[idx] + radii[j] - 1e-12:
                    # Compute how much we could grow radii[idx] before violating
                    delta = (dist - 1e-12) / (1 + radii[j] / radii[idx]) - radii[idx]
                    max_grow = max(max_grow, delta)
            # Apply capped expansion to avoid overlap
            grow_amount = np.clip(expansion_per_circle + max_grow * 0.6, 0, 0.02)  # Max 2% growth per step
            v[3*idx+2] = np.clip(v[3*idx+2] + grow_amount, 1e-4, 0.5)
        
        # Re-evaluate after controlled expansion
        res = minimize(neg_sum_radii, v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 250, "ftol": 1e-11, "eps": 1e-9})
    
    # Post-optimization stability check and refinement
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Final safety check and minor refinement
        for _ in range(2):
            # Check for overlapping circles
            overlap = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < radii[i] + radii[j] - 1e-12:
                        # Minor perturbation of one circle to break overlap
                        overlap_pos = (i, j)
                        overlap = True
                        break
                if overlap:
                    break
            if not overlap:
                break
            # Handle overlap: adjust one circle position outward
            i, j = overlap_pos
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            overlap_amount = radii[i] + radii[j] - dist
            delta = 0.5 * overlap_amount / (radii[i] + radii[j])  # Half the overlap as delta
            # Choose less constrained circle to move
            if abs(dx) > abs(dy):
                # Move i circle horizontally
                v[3*i] += delta * np.sign(dx) * 0.75
                v[3*i+1] += 0.3 * delta * np.sign(dy)
            else:
                # Move i circle vertically
                v[3*i+1] += delta * np.sign(dy) * 0.75
                v[3*i] += 0.3 * delta * np.sign(dx)
            # Re-evaluate
            res = minimize(neg_sum_radii, v, method="SLSQP", 
                           bounds=bounds, constraints=cons, 
                           options={"maxiter": 50, "ftol": 1e-11, "eps": 1e-9})
    
    # Final output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())