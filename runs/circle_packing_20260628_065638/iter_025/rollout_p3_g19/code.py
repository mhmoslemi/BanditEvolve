import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Seed for deterministic initialization
    np.random.seed(42)
    
    # Optimized initialization with improved grid configuration
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add spatial perturbation to break symmetry and improve packing
        x = x_center + np.random.uniform(-0.2, 0.2) / (cols + 1)
        y = y_center + np.random.uniform(-0.2, 0.2) / (rows + 1)
        # Staggered grid for alternate rows
        if row % 2 == 1:
            x += 0.3 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        # Left bound constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right bound constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom bound constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top bound constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Use vectorized overlap constraint function with lambda capture
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-9})

    # Implement radical geometric hashing and topological reconfiguration
    if res.success:
        v = res.x
        # Apply geometric hashing with spatial perturbation
        spatial_hash = np.random.rand(n, 2) * 0.15 - 0.075  # Balanced perturbation
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Create new constraints with adjacency relationships enforced
        def create_constraints(v):
            new_cons = []
            for i in range(n):
                new_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
                new_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
                new_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
                new_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
            for i in range(n):
                for j in range(i + 1, n):
                    def constraint_func(v, i=i, j=j):
                        dx = v[3*i] - v[3*j]
                        dy = v[3*i+1] - v[3*j+1]
                        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                    new_cons.append({"type": "ineq", "fun": constraint_func})
            return new_cons
        
        # Re-evaluate with perturbed parameters and new constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=create_constraints(perturbed_v), options={"maxiter": 1200, "ftol": 1e-11, "eps": 1e-8})
    
    # Targeted expansion with adjacency-driven configuration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by minimizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor to increase least constrained radius
        total_sum = np.sum(radii)
        expansion_factor = 0.008 / (n - 1) * 1.2  # More aggressive controlled expansion
        
        # Apply expansion with soft enforcement and gradient adjustment
        # Enforce minimum radius during expansion
        expansion_radius = min(radii) * 0.2 + 1e-5
        
        # Calculate potential expansion while preserving minimum radius
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor
        
        # Create expanded decision vector
        expanded_v = v.copy()
        expanded_v[2::3] = np.clip(new_radii, 1e-5, 0.5)
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=create_constraints(expanded_v), options={"maxiter": 900, "ftol": 1e-11, "eps": 1e-8})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())