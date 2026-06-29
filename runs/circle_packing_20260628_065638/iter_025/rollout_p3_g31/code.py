import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with dense, shuffled grid to induce complex interactions
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        x = base_x + np.random.uniform(-0.12, 0.12)
        y = base_y + np.random.uniform(-0.12, 0.12)
        xs.append(x)
        ys.append(y)
    
    # Adaptive radius initialization based on layout
    r0 = 0.45 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Fast, vectorized boundary constraints with closure capture
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized, efficient overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    if res.success:
        v = res.x
        # Implement radical spatial reconfiguration with geometric hashing
        random_hash = np.random.rand(n, 2) * 0.16 - 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        
        # Reconstruct constraints for new configuration
        def get_all_constraints(v):
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
            return cons
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=get_all_constraints(perturbed_v), options={"maxiter": 500, "ftol": 1e-12})
    
    if res.success:
        v = res.x
        # Compute distances and find the least constrained circle
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Apply radical radius expansion on least constrained circle
        radii = v[2::3]
        total_sum = np.sum(radii)
        expansion_factor = 0.012 / (n - 1)  # Controlled controlled expansion
        
        # Calculate the minimal valid expansion for the least constrained circle
        new_radii = radii.copy()
        # Estimate how much we can increase without violating constraints
        max_possible_radius = 0.5 - np.max(distances_to_edges(v, n))
        new_radii[least_constrained_idx] = np.clip(new_radii[least_constrained_idx] + expansion_factor * 1.3, 1e-4, max_possible_radius)
        
        # Apply controlled expansion to all other circles
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] = np.clip(new_radii[i] + expansion_factor * 0.95, 1e-4, max_possible_radius)
        
        # Re-evaluate with expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=get_all_constraints(v_new), options={"maxiter": 500, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())

def distances_to_edges(v, n):
    distances = np.zeros(n)
    for i in range(n):
        x = v[3*i]
        y = v[3*i+1]
        r = v[3*i+2]
        # Left edge
        distances[i] = max(0, (x - r))
        # Right edge
        distances[i] = max(distances[i], 1 - (x + r))
        # Bottom edge
        distances[i] = max(distances[i], (y - r))
        # Top edge
        distances[i] = max(distances[i], 1 - (y + r))
    return distances