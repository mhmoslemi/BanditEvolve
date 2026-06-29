import numpy as np

def run_packing():
    n = 26
    
    # Define grid parameters
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Initialize positions with staggered grid and controlled randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add controlled randomness to avoid symmetry issues
        dx = np.random.uniform(-0.04, 0.04)
        dy = np.random.uniform(-0.04, 0.04)
        x = x_center + dx
        y = y_center + dy
        
        # Create staggered grid by shifting alternate rows
        if row % 2 == 1:
            x += 0.3 / cols
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimate based on grid structure
    r0 = 0.45 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply shake heuristic: perturb small circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for all circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Identify small circles
        small_indices = np.argsort(radii)[:5]
        
        # Perform structured perturbations with spatial awareness
        for idx in small_indices:
            x, y, r = centers[idx]
            
            # Calculate direction of motion to minimize overlap
            interaction = np.sum(1 / (dists[idx] + 1e-8))
            directions = np.zeros((n, 2))
            for j in range(n):
                if j != idx:
                    dx = centers[j, 0] - x
                    dy = centers[j, 1] - y
                    directions[j] = np.array([dx, dy])
            
            # Normalize directions to get average constraint direction
            constraint_dir = np.sum(directions, axis=0)
            constraint_dir /= np.linalg.norm(constraint_dir)
            
            # Move circle in the direction of least constraint
            v[3*idx] += constraint_dir[0] * 0.005
            v[3*idx+1] += constraint_dir[1] * 0.005
            v[3*idx+2] += 0.001
        
        # Re-evaluate with perturbed configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Final optimization with additional stabilization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for all circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Calculate constraint strength per circle
        constraint_strength = np.sum(1 / (dists + 1e-8), axis=1)
        
        # Target the circle with highest constraint strength for expansion
        constrained_idx = np.argmax(constraint_strength)
        
        # Calculate expansion factor to increase this circle's radius
        total_sum = np.sum(radii)
        expansion_factor = 0.006 / (n - 1)  # Controlled expansion to unlock new configuration
        
        # Adjust radii
        new_radii = radii.copy()
        new_radii[constrained_idx] += expansion_factor * 1.5
        for i in range(n):
            if i != constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with tightened tolerances
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())