import numpy as np

def run_packing():
    n = 26
    # Introduce more sophisticated column distribution for better utilization
    cols = 6
    cols_ratio = 0.9  # Ensure that column spacing is optimized with grid spacing
    rows = (n + cols - 1) // cols
    grid_width = 1.0 / cols + 1e-6  # Slight spacing to avoid edge cases
    grid_height = 1.0 / rows + 1e-6
    
    # Introduce spatial perturbation with adaptive scaling for geometric hashing
    # This avoids symmetry locking and enhances convergence diversity
    perturbation_strength = 0.035
    
    # Initialize with randomized staggered grid with adaptive spacing and spatial hashing
    xs = []
    ys = []
    circle_weights = []  # Used for post-hoc radius expansion weighting
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Apply spatial perturbation with adaptive scaling based on grid cell size
        x = base_x + np.random.uniform(-0.03, 0.03) * (grid_width / 2)
        y = base_y + np.random.uniform(-0.03, 0.03) * (grid_height / 2)
        
        # Staggered grid adjustment
        if row % 2 == 0:
            x += 0.5 * grid_width / 2.5
        xs.append(x)
        ys.append(y)
        
        # Assign weight based on distance from center and spacing - closer circles get smaller weights
        weight = 1.0 - (np.sqrt(base_x**2 + base_y**2)) * (1.5 / (cols/2))
        circle_weights.append(weight)
    
    r0 = (0.32 / cols) * (1.0 / (1.0 + cols * 0.1)) - 1e-3  # Slightly smaller base radius for better expansion
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Create bounds with enhanced spatial sensitivity for radius expansion control
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n elements, matches v
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Minimize negative sum radii for maximization
    
    # Vectorized bounds constraints with enhanced handling
    cons = []
    for i in range(n):
        # Left bound (x - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right bound (x + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom bound (y - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top bound (y + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized pairwise distance constraint with optimized vectorization
    # Use numpy broadcasting to calculate all pairwise distances
    constraint_func_cache = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            constraint_func_cache.append(constraint_func)
    # Append to constraints list
    for f in constraint_func_cache:
        cons.append({"type": "ineq", "fun": f})
    
    # Initial optimization with adaptive tolerances and iterative refinement
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-11, "gtol": 1e-10})
    
    # Introduce phase 1: geometric dissection on most interacting pair (with heuristic selection)
    if res.success:
        # Calculate pairwise distances with vectorized broadcasting (vectorized)
        dx = np.expand_dims(v0[0::3], axis=1) - np.expand_dims(v0[0::3], axis=0)
        dy = np.expand_dims(v0[1::3], axis=1) - np.expand_dims(v0[1::3], axis=0)
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find pairwise distances that are smallest due to potential for dissection
        # Select top 5-10 pairs for dissection based on minimal pairwise distance and small radii
        candidate_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                if i != j:
                    if dists[i,j] < (v0[3*i+2] + v0[3*j+2]) + 1e-8:
                        candidate_pairs.append((i,j))
        
        # Sort candidate pairs based on the degree of tightness between centers and radii
        candidate_pairs.sort(key=lambda pair: dists[pair[0], pair[1]] - (v0[3*pair[0]+2] + v0[3*pair[1]+2]))
        
        # Select top 3 most interacting pairs for targeted dissection
        # Use a novel approach to force geometric dissection on these pairs
        if len(candidate_pairs) >= 3:
            pair1, pair2, pair3 = candidate_pairs[0], candidate_pairs[1], candidate_pairs[2]
            
            # Extract coordinates for dissection
            x1, y1 = v0[3*pair1[0]], v0[3*pair1[1]]
            y1_idx = 3*pair1[1]+1
            r1 = v0[3*pair1[0]+2]
            r2 = v0[3*pair1[1]+2]
            
            x2, y2 = v0[3*pair2[0]], v0[3*pair2[1]]
            y2_idx = 3*pair2[1]+1
            r3 = v0[3*pair2[0]+2]
            r4 = v0[3*pair2[1]+2]
            
            x3, y3 = v0[3*pair3[0]], v0[3*pair3[1]]
            y3_idx = 3*pair3[1]+1
            r5 = v0[3*pair3[0]+2]
            r6 = v0[3*pair3[1]+2]
            
            # Create a geometric dissection matrix that forces reordering
            # Define transformation matrix and new spatial hashing
            displacement = 0.02  # Displacement vector to reconfigure spatial relationships
            scale_factor = 1.2  # Scale to increase spacing between dissection pairs
            
            # Create geometric dissection vector for the selected pairs
            dissection_v = v0.copy()
            dissection_v[3*pair1[0]] = x1 + displacement * scale_factor
            dissection_v[3*pair1[1]] = x2 - displacement * scale_factor
            dissection_v[3*pair2[0]] = x2 + displacement * scale_factor
            dissection_v[3*pair2[1]] = x3 - displacement * scale_factor
            dissection_v[3*pair3[0]] = x3 + displacement * scale_factor
            dissection_v[3*pair3[1]] = x1 - displacement * scale_factor
            
            # Apply a novel geometric hashing to ensure convergence diversity
            perturbation = np.random.rand(n,2) * 0.015
            perturbed_v = dissection_v.copy()
            for idx in range(n):
                perturbed_v[3*idx] += perturbation[idx,0]
                perturbed_v[3*idx + 1] += perturbation[idx,1]
            
            # Re-optimize with new spatial configuration and enhanced tolerances
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-12})
    
    # Introduce phase 2: controlled radius expansion on the least constrained circle with topology reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute all pairwise distances with vectorized broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute distance to others for each circle
        min_dist = np.min(dists, axis=1)
        min_dist_idx = np.argmin(min_dist)
        
        # Select the circle with the most space (lowest min distance) to expand
        expansion_circle_idx = min_dist_idx
        
        # Calculate the initial radius expansion potential
        base_growth = 0.013
        # Apply weight-based expansion to the least constrained circle
        expansion_amount = base_growth * circle_weights[expansion_circle_idx] / np.sum(circle_weights)
        # Introduce small randomness for enhanced exploration
        expansion_amount += 0.0005 * np.random.randn()
        
        # Create expansion vector and apply expansion to all circles with slight prioritization
        new_radii = np.clip(radii + (expansion_amount * 1.1), 1e-6, 0.45)
        
        # Perturbation to avoid local optima
        spatial_hash = np.random.rand(n,2) * 0.01
        for idx in range(n):
            # Adjust spatial positions with small perturbations based on radii scaling
            v[3*idx] += spatial_hash[idx,0] * (new_radii[idx] / np.mean(new_radii)) * 1.2
            v[3*idx + 1] += spatial_hash[idx,1] * (new_radii[idx] / np.mean(new_radii)) * 1.2
        
        # Final optimization pass with enhanced constraints and tolerances
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 350, "ftol": 1e-12, "gtol": 1e-13})
    
    # Apply post-expansion refinement with adaptive constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Final boundary enforcement with tighter tolerances
        for i in range(n):
            if v[3*i] - radii[i] < -1e-12 or v[3*i] + radii[i] > 1 + 1e-12:
                v[3*i] = max(min(v[3*i], 1.0), 0.0)
            if v[3*i+1] - radii[i] < -1e-12 or v[3*i+1] + radii[i] > 1 + 1e-12:
                v[3*i+1] = max(min(v[3*i+1], 1.0), 0.0)
        # Final radius clipping and enforcement
        v[2::3] = np.clip(v[2::3], 1e-6, 0.45)
        
        # Refinement pass to ensure validity
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 100, "ftol": 1e-13, "gtol": 1e-14})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())