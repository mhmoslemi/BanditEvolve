import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Create geometrically balanced initial positions with spatial randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid: (col+0.5)/cols, (row+0.5)/rows
        grid_x = (col + 0.5) / cols
        grid_y = (row + 0.5) / rows
        
        # Spatial hashing with adaptive scale to prevent symmetry lock
        hash_x = np.random.rand() * 0.08
        hash_y = np.random.rand() * 0.08
        
        # Add spatial randomness weighted by radius potential
        x = grid_x + hash_x * (0.8 / cols) * (1.0 / (n**(1/3)))
        y = grid_y + hash_y * (0.8 / rows) * (1.0 / (n**(1/3)))
        
        # Stagger alternate rows with dynamic offset
        row_offset = 0.25 / (cols * (n**0.25))
        if row % 2 == 1:
            x += row_offset
        
        xs.append(x)
        ys.append(y)
    
    # Initialize radius with optimized base value
    r0 = 0.45 / cols * (1.0 + np.random.rand() * 0.1) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []

    # Create boundary constraints with adaptive tolerance
    for i in range(n):
        # Left + radius <= 1.0 (with dynamic scaling factor)
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2] + 1e-6 * (v[3*i+2] ** 0.5) if v[3*i+2] > 0 else float('inf')})
        # Right - radius >= 0 (with dynamic scaling)
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2] - 1e-6 * (v[3*i+2] ** 0.5) if v[3*i+2] > 0 else float('inf')})
        # Bottom + radius <= 1.0 (same treatment)
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2] + 1e-6 * (v[3*i+2] ** 0.5) if v[3*i+2] > 0 else float('inf')})
        # Top - radius >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2] - 1e-6 * (v[3*i+2] ** 0.5) if v[3*i+2] > 0 else float('inf')})

    # Create optimized non-overlap constraints with exponential scaling
    for i in range(n):
        for j in range(i+1, n):
            # Overlap constraint with adaptive weight (higher weight for closer circles)
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r_sum = v[3*i+2] + v[3*j+2]
                # Add exponential weight based on current radii
                weight = 1.0 + (r_sum / (0.1 * n)) * (1.0 / (dist_sq ** 0.2 if dist_sq > 0 else 1.0))
                return dist_sq - (r_sum ** 2) + 1e-8 * weight
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization with improved convergence and exploration
    res1 = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                    constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-9})

    # First refinement with spatial hashing and perturbation
    if res1.success:
        v = res1.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hashing matrix with adaptive scale
        hash_matrix = np.random.rand(n, 2) * 0.04 * (1.0 / (np.std(radii) * 1.5))
        
        # Perturb positions with spatial hashing matrix
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_matrix[i, 0] * radii[i] * (1.0 / np.median(radii))
            perturbed_v[3*i+1] += hash_matrix[i, 1] * radii[i] * (1.0 / np.median(radii))
        
        # Refine with enhanced optimization
        res2 = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                        constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})
        
        # Second refinement with targeted radius expansion and symmetry-breaking
        if res2.success:
            v = res2.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            dist_mat = np.zeros((n, n))
            
            # Efficient distance calculation with broadcasting
            dx_vec = centers[np.newaxis, :, 0] - centers[:, np.newaxis, 0]
            dy_vec = centers[np.newaxis, :, 1] - centers[:, np.newaxis, 1]
            dist_mat = np.sqrt(dx_vec**2 + dy_vec**2)
            
            # Find circle with largest non-overlap margin (least constrained)
            min_dists = np.min(dist_mat, axis=1)
            least_constrained_idx = np.argmax(min_dists)
            
            # Calculate base expansion and apply it in a dynamic way
            current_total = np.sum(radii)
            target_expansion_ratio = 0.0085
            target_sum = current_total * (1.0 + target_expansion_ratio)
            
            # Dynamic expansion vector
            new_radii = radii.copy()
            expansion_factor = (target_sum - current_total) / (n - 1)
            
            # Use stochastic expansion with radius-based scale
            for i in range(n):
                if i != least_constrained_idx:
                    # Introduce slight randomness for diversity but constrained
                    expansion_i = expansion_factor * (1.0 + (np.random.rand() * 0.1) * (radii[i]/np.std(radii)))
                    new_radii[i] += expansion_i
            
            # Create new decision vector
            new_v = v.copy()
            new_v[2::3] = new_radii
            
            # Final optimization with enhanced constraints and exploration
            res3 = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                            constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9})
        
    # Fall back to previous solution if all optimizations failed
    v = res1.x if res3 is None or not res3.success else res3.x
    v = res2.x if (res3 is None or not res3.success) and res2.success else v
    v = res1.x if not res2.success else v
    
    # Final cleanup
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    return centers, radii, float(radii.sum())