import numpy as np

def run_packing():
    n = 26
    
    # Strategic initial seeding with spatial partitioning and probabilistic bias
    # We'll use an optimized grid with dynamic rows and more robust randomized seeding
    cols = min(5, int(np.ceil(np.sqrt(n * 2))))  # Optimize row count for 26 circles
    rows = (n + cols - 1) // cols
    
    # Initialize centers with randomized geometric perturbation and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid positions (adjusted for staggered grid)
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Compute randomization factor for bias towards corners
        bias_factor = 0.1 + 0.2 * (0.5 - abs(row - rows/2))  # Bias to edges
        randomization = np.random.uniform(-bias_factor, bias_factor, size=2)
        
        # Stagger alternate rows to prevent vertical clustering
        stagger = (row % 2 == 1) * 0.5 / cols
        
        x = base_x + randomization[0] + stagger
        y = base_y + randomization[1]
        
        # Apply non-uniform constraint to x-axis for better edge utilization
        x = np.clip(x, 0.1, 0.9)
        y = np.clip(y, 0.1, 0.9)
        
        xs.append(x)
        ys.append(y)
    
    # Optimized initial radius estimate with edge expansion
    r0 = 0.30 / cols - 1e-3 * (1.2 if n > 20 else 1.0)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define strict bounds with precise length constraint check
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n elements for 3 per circle

    # Objective function with regularization
    def neg_sum_radii(v):
        r = v[2::3]
        # Add regularization to prevent all circles from shrinking/overexpanding
        return -np.sum(r) + 1e-4 * np.abs(np.diff(r)).sum() + 1e-3 * np.sum(r ** 2)

    # Vectorized constraints with optimized closure handling
    cons = []

    # Boundary constraints with optimized lambda capturing
    for i in range(n):
        # Left edge: x - r >= 0 → x - r >= 0 → x >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right edge: 1 - x - r >= 0 → x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom edge: y - r >= 0 → y >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top edge: 1 - y - r >= 0 → y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints with spatial hashing and gradient approximation optimization
    # Use vectorized calculation with explicit parameter passing to prevent lambda closure issues
    for i in range(n):
        for j in range(i + 1, n):
            # Use more precise squared distance calculation
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization with enhanced control
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-12,
                                              "disp": False})

    # Advanced reconfiguration with spatial hashing of constraint gradients
    if res.success:
        # Apply adaptive spatial gradient hashing for reconfiguration
        centers = np.column_stack([res.x[0::3], res.x[1::3]])
        radii = res.x[2::3]
        
        # Generate spatial gradient perturbation matrix with adaptive scaling
        grad_hashes = np.random.rand(n, 2) * ((radii / radii.mean()) ** 0.7)
        perturbed_v = res.x.copy()
        
        # Apply staggered perturbation to edge circles for better boundary utilization
        for i in range(n):
            if abs(centers[i,0] - 0.5) < 0.25 or abs(centers[i,1] - 0.5) < 0.25:
                perturbed_v[3*i] += grad_hashes[i, 0] * np.sin(np.pi / 4)
                perturbed_v[3*i+1] += grad_hashes[i, 1] * np.cos(np.pi / 4)
            else:
                perturbed_v[3*i] += grad_hashes[i, 0]
                perturbed_v[3*i+1] += grad_hashes[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11,
                                                  "eps": 1e-12, "disp": False})
    
    # Apply asymmetric reconfiguration targeting the most underutilized circle
    if res.success:
        centers = np.column_stack([res.x[0::3], res.x[1::3]])
        radii = res.x[2::3]
        
        # Compute distance matrix with optimized broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify the most underutilized circle by max distance from edge
        edge_distances = np.min(np.stack([centers[:, 0] - radii, 
                                          1 - centers[:, 0] - radii,
                                          centers[:, 1] - radii,
                                          1 - centers[:, 1] - radii]), axis=0)
        underutilized_idx = np.argmin(edge_distances)
        
        # Introduce asymmetric expansion to underutilized circle with edge bias
        expansion_factor = 0.006 * (1.0 + 0.2 * (1 - np.min(edge_distances)))
        # Introduce stochastic bias toward unoccupied space edges
        expansion_dir = np.random.choice([ [0, 1], [1, 0], [0, -1], [-1, 0] ], size=1)[0]
        expansion_dir = np.array(expansion_dir) / np.linalg.norm(expansion_dir)
        
        # Apply asymmetric expansion to center of underutilized circle
        under_center = centers[underutilized_idx]
        under_radius = radii[underutilized_idx]
        
        # Generate new centers with expansion
        new_centers = centers.copy()
        new_centers[underutilized_idx] = under_center + expansion_dir * (under_radius * 1.1)
        
        # Compute new radii with gradient-preserving adjustment
        new_radii = radii.copy()
        new_radii[underutilized_idx] += expansion_factor * (1.0 + 0.3 * np.random.rand())
        
        # Validate and re-optimize
        while True:
            expanded_v = res.x.copy()
            expanded_v[0::3] = new_centers[:, 0]
            expanded_v[1::3] = new_centers[:, 1]
            expanded_v[2::3] = new_radii
            
            # Perform fast validation
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_v[3*i] - expanded_v[3*j]
                    dy = expanded_v[3*i+1] - expanded_v[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion gradually to maintain validity
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final optimization with adjusted configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11,
                                                  "eps": 1e-12, "disp": False})

    # Final clean-up to ensure numerical stability
    v = res.x if res.success else res.x
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())