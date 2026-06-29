import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Implement randomized geometric tiling with enhanced stochasticity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Add more diverse randomness for better spatial distribution
        x = base_x + np.random.uniform(-0.07, 0.07)
        y = base_y + np.random.uniform(-0.07, 0.07)
        # Staggered row offset with adaptive magnitude
        if row % 2 == 1:
            x += 0.5 / cols * (0.3 + np.random.uniform(-0.05, 0.05))
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with better closure handling
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized pairwise overlap constraints with closure optimization
    overlap_constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            overlap_constraints.append({"type": "ineq", "fun": constraint_func})

    cons.extend(overlap_constraints)

    # First-stage optimization with high precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Apply stochastic reset to least constrained circles for topological shift
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            dx = centers[:, 0] - centers[i, 0]
            dy = centers[:, 1] - centers[i, 1]
            dist_matrix[i, :] = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle (max min distance)
        min_dists = np.min(dist_matrix, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Apply stochastic reset to this circle and neighbors
        perturb_strength = 0.05
        new_v = v.copy()
        new_v[3*least_constrained_idx] += np.random.uniform(-perturb_strength, perturb_strength)
        new_v[3*least_constrained_idx+1] += np.random.uniform(-perturb_strength, perturb_strength)
        
        # Reset neighbors' positions with moderate perturbation
        for neighbor in [least_constrained_idx - 1, least_constrained_idx + 1, 
                         least_constrained_idx - cols, least_constrained_idx + cols]:
            if 0 <= neighbor < n:
                new_v[3*neighbor] += np.random.uniform(-0.01, 0.01)
                new_v[3*neighbor+1] += np.random.uniform(-0.01, 0.01)
        
        # Re-optimize with new configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})

    # Targeted expansion of least constrained circle with geometric awareness
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            dx = centers[:, 0] - centers[i, 0]
            dy = centers[:, 1] - centers[i, 1]
            dist_matrix[i, :] = np.sqrt(dx**2 + dy*2)
        
        # Find least constrained circle with geometric awareness
        min_dists = np.min(dist_matrix, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate maximum possible expansion with geometric constraint
        max_possible_expansion = 0.002
        expanded_radius = radii[least_constrained_idx] + max_possible_expansion
        
        # Compute expansion factor
        total_sum = np.sum(radii)
        expansion_factor = (total_sum + max_possible_expansion) / n
        
        # Adjust radii for expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] = expanded_radius
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] = min(radii[i] + expansion_factor, 0.5)
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with tightened tolerances
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 700, "ftol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())