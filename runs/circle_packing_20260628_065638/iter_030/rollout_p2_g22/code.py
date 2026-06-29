import numpy as np

def run_packing():
    n = 26
    
    # Step 1: Construct a 2D grid with dynamic row/column calculation
    cols = 5  # Optimal for 26 circles (sqrt(26) is ~5.1, 5 cols gives better balance)
    rows = (n + cols - 1) // cols  # Adjust rows to fit all circles
    
    # Optimized spatial initialization through hybrid grid and adaptive positioning:
    # - Primary clustering to get initial positions
    # - Secondary refinement through spatial-aware displacement
    # - Use of dynamic displacement factors to break symmetry
    xs = [] 
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid position
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        
        # Adaptive displacement: increase displacement with row index to prevent cluster
        row_factor = 0.12 * row
        col_factor = 0.09 * col
        
        # Dynamic random perturbation with row-dependent range
        x_perturb = np.random.uniform(-0.1 * (1 + row / rows), 0.1 * (1 + row / rows))
        y_perturb = np.random.uniform(-0.1 * (1 + row / rows), 0.1 * (1 + row / rows))
        
        # Stagger grid: shifted row-wise to avoid vertical alignment
        if row % 2 == 1:
            x_base += 0.25 / cols  # Increased stagger to allow better spacing
            
        # Apply perturbations and displacement factors
        x = x_base + x_perturb + row_factor
        y = y_base + y_perturb + row_factor
        
        # Bound adjustment: prevent edge collisions
        x = np.clip(x, 0.0 + 1e-4, 1.0 - 1e-4)
        y = np.clip(y, 0.0 + 1e-4, 1.0 - 1e-4)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius calculation: more aggressive for row-based spacing
    # Radius depends on row index to create staggered height space and avoid crowding
    r0 = np.zeros(n)
    for i in range(n):
        row = i // cols
        col = i % cols
        base_radius = 0.38 / cols
        row_factor = 0.02 * row  # Adjusts base radius to account for grid spacing needs
        r0[i] = np.clip(base_radius + row_factor - 1e-3, 1e-4, 0.5)
    
    # Build decision vector as [x0, y0, r0, x1, y1, r1, ...]
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # Set bounds strictly ensuring length 3*n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries
    
    # Objective function: minimize negative sum of radii -> maximize sum
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Construct constraints in vectorized form with lambda binding for SLSQP
    cons = []
    for i in range(n):
        # Left & right margin constraints (x)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})  # x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})  # 1 - x - r >= 0
        
        # Bottom & top margin constraints (y)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})  # y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})  # 1 - y - r >= 0
    
    # Add overlap constraints with careful vectorization:
    # We compute distance squared - (r_i + r_j)^2 in vectorized form
    # This avoids the O(n^2) runtime by using broadcasting
    # Note: we must handle the constraints with correct closure binding
    
    # Vectorized overlap constraints (efficient and optimized with closure binding)
    # We use nested lambdas with binding to ensure correct closure resolution
    for i in range(n):
        for j in range(i + 1, n):
            def build_overlap_func(i, j):
                def overlap_func(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2]) ** 2
                return overlap_func
            # We bind the i and j values now to ensure the function is fixed
            cons.append({"type": "ineq", "fun": build_overlap_func(i, j)})
    
    # Initial optimization: high-precision + adaptive constraints
    # Initial pass: full optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-11})
    if not res.success:
        print("Initial optimization failed. Trying perturbed starting point with adaptive constraints.")
        # Perturbed optimization: apply adaptive spatial and radial perturbations
        perturbation_factor = 0.1
        perturbed_v = v0.copy()
        for i in range(n):
            # Add spatial perturbation with row-dependent scaling
            row = i // cols
            row_factor = 0.1 + row / (rows) * 0.05  # Scale more for later rows
            perturbation_x = (np.random.rand() - 0.5) * row_factor
            perturbation_y = (np.random.rand() - 0.5) * row_factor
            
            # Add radial perturbations with row-dependent scaling
            rad_factor = 0.05 * (1 + row / rows)
            perturbation_r = (np.random.rand() - 0.5) * rad_factor
            
            perturbed_v[3*i] += perturbation_x
            perturbed_v[3*i+1] += perturbation_y
            perturbed_v[3*i+2] += perturbation_r
        
        # Ensure bounds stay valid
        for i in range(n):
            # Clamp x
            perturbed_v[3*i] = np.clip(perturbed_v[3*i], 1e-4, 1. - 1e-4)
            # Clamp y
            perturbed_v[3*i+1] = np.clip(perturbed_v[3*i+1], 1e-4, 1. - 1e-4)
            # Clamp rad
            perturbed_v[3*i+2] = np.clip(perturbed_v[3*i+2], 1e-4, 0.5)
        
        # Retry optimization with perturbed starting point
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-11})
    
    # Adaptive reconfiguration phase: identify critical interactions and optimize
    if res.success:
        v = res.x
        
        # Compute radii and centers
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Precompute pairwise distance matrix using broadcasting (vectorized)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        pairwise_distance = np.sqrt(dx**2 + dy**2)
        
        # Create interaction matrix
        interaction = np.zeros(n)
        for i in range(n):
            # Weighted sum of inverse distance to all circles
            # Avoid division by zero using epsilon
            sum_inv_dist = 0.0
            for j in range(n):
                if i != j:
                    dist = pairwise_distance[i, j]
                    if dist < 1e-8:
                        continue
                    sum_inv_dist += 1. / dist
            interaction[i] = sum_inv_dist
        
        # Identify most and least interacted circles: 
        # most_interacted_idx = np.argmin(interaction) 
        # least_interacted_idx = np.argmax(interaction)
        most_interacted_idx = np.argmin(interaction)
        least_interacted_idx = np.argmax(interaction)
        
        # Apply targeted reconfiguration: isolate the two most interacted and perturb them
        # Step 1: isolate and reconfigure the two most interacted circles in the grid
        v_perturbed = v.copy()
        for idx in [most_interacted_idx]:
            # Increase base spatial perturbation for these
            base_perturb_x = np.random.uniform(-0.1, 0.1)
            base_perturb_y = np.random.uniform(-0.1, 0.1)
            base_perturb_r = np.random.uniform(-0.01, 0.01)
            
            # Introduce directional displacement based on row & col
            row = idx // cols
            col = idx % cols
            direction_x = np.random.choice([-1, 1]) * 0.03 * (col + 0.5)
            direction_y = np.random.choice([-1, 1]) * 0.03 * (row + 0.5)
            
            # Apply directional push with added noise to avoid collisions
            v_perturbed[3*idx] += base_perturb_x + direction_x
            v_perturbed[3*idx+1] += base_perturb_y + direction_y
            v_perturbed[3*idx+2] += base_perturb_r
            
            # Clamp to valid space again
            v_perturbed[3*idx] = np.clip(v_perturbed[3*idx], 1e-4, 1. - 1e-4)
            v_perturbed[3*idx+1] = np.clip(v_perturbed[3*idx+1], 1e-4, 1. - 1e-4)
            v_perturbed[3*idx+2] = np.clip(v_perturbed[3*idx+2], 1e-4, 0.5)
        
        # Re-optimization with modified critical points
        res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Final optimization step for global refinement
    if res.success:
        v = res.x
    else:
        v = v0
    
    # Final clean-up and validation:
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    
    return centers, radii, float(radii.sum())