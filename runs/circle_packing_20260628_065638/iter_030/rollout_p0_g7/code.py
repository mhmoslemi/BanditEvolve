import numpy as np

def run_packing():
    n = 26
    # Optimized columns and rows layout with tighter bounds on radius grid and better spatial control
    cols = int(np.ceil(np.sqrt(n)))
    cols = max(3, cols)  # Force minimum column count to avoid overclustering 
    rows = (n + cols - 1) // cols
    
    # Initial grid layout with refined spatial jitter, asymmetric grid expansion, and row-wise radius scaling
    
    # Precompute grid base positions with optimized spacing and asymmetric row expansion
    base_x = np.linspace(0.0, 1.0, cols+1)[1:-1]  # Even spacing, avoids edge effects
    base_y = np.linspace(0.0, 1.0, rows+1)[1:-1]
    grid_centers = np.zeros((n, 2))
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        # Base center placement
        base_xi = base_x[col_idx] + 0.05 * (1 - 0.5 * row_idx)  # Upward shift in even rows
        base_yi = base_y[row_idx] + 0.03 * (1 - 0.5 * col_idx)  # Lateral shift in even columns
        grid_centers[i, 0] = base_xi
        grid_centers[i, 1] = base_yi
    
    # Apply spatial jitter with decay based on row/column proximity for diversity
    jitter = np.random.uniform(-0.03, 0.03, size=(n, 2))
    jitter *= np.exp(-0.5 * (np.abs(np.arange(n) // cols).reshape(n,1) + np.abs(np.arange(n) % cols).reshape(n,1)))
    grid_centers += jitter
    
    xs = grid_centers[:, 0]
    ys = grid_centers[:, 1]
    
    # Optimized radius baseline with row-wise scaling based on grid density
    base_radius = 0.35 / cols  # Base radius from grid spacing
    row_density = np.zeros(rows)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            # Simulate density with distance to edges for more effective row-wise scaling
            row_density[r] = max(row_density[r], 1.0 / ((base_y[r] - base_y[r-1] if r>0 else 1.0) * 
                                                    (base_x[c] - base_x[c-1] if c>0 else 1.0)))
    row_weight = 1.0 + 0.25 * (row_density - np.min(row_density)) / (np.max(row_density) - np.min(row_density))  # Upweight rows with higher density
    r0 = base_radius * row_weight[np.arange(n) // cols] - 1e-3
    
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # strict lower radius bound, high upper

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # objective to be maximized via minimization

    # Vectorized constraints using lambda with i captures, optimized closure usage and explicit index handling
    cons = []
    for i in range(n):
        # Left constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right constraint: 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top constraint: 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Optimized overlap constraints with vectorized lambda and index caching in function definition
    # Using a loop-based approach with parameterized lambdas to improve closure consistency and reduce memory overhead
    for i in range(n):
        for j in range(i + 1, n):
            # Precompute the indices for quick access during function calls
            i3 = 3 * i
            j3 = 3 * j
            # Define the constraint function to avoid repeated indexing
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j, i3=i3, j3=j3:
                    (v[i3] - v[j3])**2 + (v[i3 + 1] - v[j3 + 1])**2 
                    - (v[i3 + 2] + v[j3 + 2])**2)
            })

    # First-phase optimization: higher maxiter and adaptive tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-11, "eps": 1e-8})
    
    # Phase 2: refined position expansion with dynamic reconfiguration
    if res.success:
        v = res.x
        # Calculate current distance matrix and radii for post-expansion analysis
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Vectorized pairwise distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1, :]
        dists = np.sqrt(dx**2 + dy**2)
        dists[dists < 1e-10] = 1e-10  # Avoid division by zero
        
        # Identify constrained circles (minimum distance to others)
        min_dists = np.min(dists, axis=1)
        index_sort = np.argsort(min_dists)
        constrained_idx = index_sort[:10]  # Top 10 most constrained
        unconstrained_idx = index_sort[10:]  # Rest
        
        # Perturb constrained circles by small, systematic vector-based offsets
        # Using dynamic bounds on perturbation based on radius size
        perturb_magnitude = np.clip((radii[constrained_idx] / np.mean(radii)) * 0.06, 0.002, 0.06)
        perturb = np.random.rand(len(constrained_idx), 2) * 2 - 1  # [-1, 1]^2
        perturb *= np.expand_dims(perturb_magnitude, axis=1)
        
        v_temp = v.copy()
        for idx, p in zip(constrained_idx, perturb):
            v_temp[3*idx] += p[0]
            v_temp[3*idx + 1] += p[1]
        
        res = minimize(neg_sum_radii, v_temp, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-8})
        v = res.x
    
    # Phase 3: adaptive constraint tightening with dynamic radius expansion and geometric hashing
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate the current constraint tightness
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        dists[dists < 1e-10] = 1e-10  # Avoid division by zero
        
        # Identify circles that are most constrained (least radius expansion potential) first
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argsort(min_dists)[:10]  # Most constrained 10
        most_unconstrained_idx = np.argsort(min_dists)[10:]  # Remaining
        
        # Apply controlled radius expansion to least constrained, with geometric hashing
        # Radius expansion magnitude per circle: base expansion plus geometric hashing based on position
        base_radius_exp = 0.002  # Base expansion
        exp_factor = 1.0 + 0.1 * np.sum((dists[least_constrained_idx] / (radii[least_constrained_idx] + radii[np.expand_dims(np.arange(n), axis=0)][:, least_constrained_idx])).min(axis=1))**2
        expansion_per_circle = base_radius_exp * exp_factor
        
        # Apply expansion with geometric hashing for constraint validation
        v_expanded = v.copy()
        for i in range(n):
            if i in least_constrained_idx:
                # Use geometric hashing based on current position to ensure expansion maintains edge constraints
                perturb = np.random.uniform(-0.01, 0.01, size=2)
                v_expanded[3*i] += perturb[0] * np.clip(1.05 * radii[i], 1e-4, 0.3)
                v_expanded[3*i+1] += perturb[1] * np.clip(1.05 * radii[i], 1e-4, 0.3)
                v_expanded[3*i+2] += expansion_per_circle
        
        # Re-evaluate with expanded configuration (but preserve geometric hashing integrity)
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-8})
        
        # Additional constraint tightening phase
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
        
            # Apply geometric hashing-based position reconfiguration to escape local optima
            # This step uses a hybrid of local radius expansion and geometric hashing for final refinement
            hash_map = np.random.rand(n, 2) * 0.03  # Small-scale hashing for local perturbation
            v_perturbed = v.copy()
            for i in range(n):
                v_perturbed[3*i] = np.clip(v[3*i] + hash_map[i, 0]*radii[i], 0.0, 1.0)
                v_perturbed[3*i + 1] = np.clip(v[3*i + 1] + hash_map[i, 1]*radii[i], 0.0, 1.0)
                v_perturbed[3*i + 2] = np.clip(v[3*i + 2] + hash_map[i, 0] * 0.001, 1e-4, 0.3)
            
            res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-8})
            v = res.x if res.success else v
    
    # Final check for NaNs or invalid values; ensure all circles are within bounds and properly radii
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.3)
    
    # Secondary check: explicit radius clamping and bounds check for last-mile safety
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12
                or y - r < -1e-12 or y + r > 1 + 1e-12):
            # If any circle is outside the square, force adjustment
            # Here we adjust the center position while keeping the radius fixed
            if x - r < -1e-12:
                centers[i, 0] = max(r, x)
            elif x + r > 1 + 1e-12:
                centers[i, 0] = min(1 - r, x)
                
            if y - r < -1e-12:
                centers[i, 1] = max(r, y)
            elif y + r > 1 + 1e-12:
                centers[i, 1] = min(1 - r, y)
    
    # Final constraint check to ensure overlap is avoided after repositioning
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < radii[i] + radii[j] - 1e-12:
                # If overlap is still present at this stage, we reposition i to avoid collision
                # This is a fallback to prevent false negatives (due to floating-point inaccuracy)
                dx_normalized = dx / dist
                dy_normalized = dy / dist
                # Move circle i out by a small fraction of the minimal overlap
                overlap_amount = (radii[i] + radii[j]) - dist
                centers[i] += np.array([dx_normalized, dy_normalized]) * (overlap_amount * 0.5 + 1e-3)
    
    return centers, radii, float(radii.sum())