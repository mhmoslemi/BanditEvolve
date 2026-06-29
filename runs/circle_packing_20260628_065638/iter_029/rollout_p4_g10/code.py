import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))  # Optimized grid resolution based on square root
    rows = (n + cols - 1) // cols  # Maintain grid-based spatial distribution
    base_grid = np.indices((rows, cols)).reshape(2, -1).T  # Canonical grid
    
    # Initialize: hybrid grid + geometric hashing + stochastic perturbation
    xs = []
    ys = []
    radii_initial = []
    
    for i in range(n):
        row = i // cols
        col = i % cols
        # Grid coordinates with geometric scaling
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Add geometric hashing based on i
        hash_offset = (i % 2 * 0.15) + (i // 13 * 0.08)
        x_hash = (np.sin(i * 0.3) + 1) * base_x * 0.8
        y_hash = (np.cos(i * 0.2) + 1) * base_y * 0.8
        
        # Stochastic spatial perturbation and staggering
        x = base_x + x_hash + np.random.uniform(-0.05, 0.05)
        y = base_y + y_hash + np.random.uniform(-0.05, 0.05)
        
        # Stagger even-odd rows for non-rectangular grid alignment
        if row % 2 == 1:
            x += 0.5 / cols * 0.65
        
        # Ensure coordinates within [0,1] with soft clipping
        x = np.clip(x, 1e-8, 1 - 1e-8)
        y = np.clip(y, 1e-8, 1 - 1e-8)
        
        xs.append(x)
        ys.append(y)
        
        # Initial radius based on grid density and geometric hashing
        radius = 0.35 / cols * 0.95 * (1 + np.sin(i * 0.12))
        radius = max(4e-3, min(radius, 0.4))  # Dynamic clamping
        radii_initial.append(radius)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.array(radii_initial)

    # Strict bounds: 3*n entries for 3*26 variables
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective to maximize radii sum

    # Constraint architecture designed for performance and stability
    cons = []

    # Constraint creation with closures capturing indices properly
    for i in range(n):
        # Left + radius <=1
        def constraint_left(v, i=i):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            return 1.0 - x - r
        cons.append({"type": "ineq", "fun": constraint_left})
        
        # Right - radius >=0
        def constraint_right(v, i=i):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            return x - r
        cons.append({"type": "ineq", "fun": constraint_right})
        
        # Bottom + radius <=1
        def constraint_bottom(v, i=i):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            return 1.0 - y - r
        cons.append({"type": "ineq", "fun": constraint_bottom})
        
        # Top - radius >=0
        def constraint_top(v, i=i):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            return y - r
        cons.append({"type": "ineq", "fun": constraint_top})
    
    # Vectorized pairwise overlap constraints with adaptive precision
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_overlap(v, i=i, j=j):
                xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                dx = xi - xj
                dy = yi - yj
                return dx*dx + dy*dy - (ri + rj)**2  
            cons.append({"type": "ineq", "fun": constraint_overlap})
    
    # Initial optimization with precise, adaptive settings
    res = minimize(neg_sum_radii, v0, 
                   method="SLSQP",
                   bounds=bounds,
                   constraints=cons,
                   options={
                       "maxiter": 2500,  # Extended for deeper exploration
                       "ftol": 1e-10,    # Very tight tolerance at convergence
                       "gtol": 1e-9,    # Gradient tolerance
                       "eps": 1e-8,     # Step size adjustment
                       "disp": False    # No verbose output
                   })
    
    # Spatial reconfiguration with geometric hashing: break symmetry
    if res.success:
        v = res.x
        # Generate dynamic spatial hashing with radius-scaled perturbations
        hash_map = np.random.rand(n, 2) * 0.04 * np.sqrt(np.sum(v[2::3]**2)) * 0.8
        perturbed_v = v.copy()
        
        # Apply spatial hashing with radius-aware scaling
        for i in range(n):
            perturbed_v[3*i] += hash_map[i, 0] * (v[3*i+2] / (0.45)) * 1.1
            perturbed_v[3*i+1] += hash_map[i, 1] * (v[3*i+2] / (0.45)) * 1.1
        
        # Reevaluate with perturbed parameters using improved constraint tightening
        res = minimize(neg_sum_radii, perturbed_v, 
                       method="SLSQP",
                       bounds=bounds,
                       constraints=cons,
                       options={
                           "maxiter": 600, 
                           "ftol": 1e-11, 
                           "gtol": 1e-10, 
                           "eps": 1e-9, 
                           "disp": False
                       })
    
    # Dynamic radius expansion through geometric optimization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized distance matrix computation with optimized broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Targeted expansion on least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Circle with minimal interactions
        target_expansion = 0.0065  # Adjusted expansion target

        # Create candidate radii vector
        candidate_radii = radii.copy()
        candidate_radii[least_constrained_idx] = min(0.42, 
                                                    candidate_radii[least_constrained_idx] + (target_expansion * 1.1))
        
        # Gradient-based expansion using directional optimization
        # Compute gradients and optimize constrained expansion
        def expansion_objective(v):
            return -np.sum(v[2::3])  # Maximize expanded radii
        
        # Initial guess using current solution
        expanded_v = v.copy()
        expanded_v[2::3] = candidate_radii
        
        # Reconfigure with expanded radii and stricter constraint enforcement
        res_expanded = minimize(expansion_objective, expanded_v,
                               method="SLSQP",
                               bounds=bounds,
                               constraints=cons,
                               options={
                                   "maxiter": 500, 
                                   "ftol": 1e-11, 
                                   "gtol": 1e-9, 
                                   "eps": 1e-8,
                                   "disp": False
                               })
        
        if res_expanded.success:
            v = res_expanded.x
            # Final post-expansion validation pass
            res = res_expanded
    
    # Apply soft clipping to ensure radii non-negative and within bounds
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.45)  # Hardcap on radii to prevent overgrowth
    return centers, radii, float(radii.sum())