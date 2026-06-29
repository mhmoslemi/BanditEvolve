import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
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

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-10})

    # First shake heuristic: small perturbation to all circles
    if res.success:
        v = res.x
        # Random shake with small offsets
        shake_amplitude = 0.002
        v_shaken = v.copy()
        v_shaken[0::3] += np.random.uniform(-shake_amplitude, shake_amplitude, size=n)
        v_shaken[1::3] += np.random.uniform(-shake_amplitude, shake_amplitude, size=n)
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v_shaken, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-10})
    
    # Targeted reconfiguration: focus on small and under-constrained circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find under-constrained and small-radius circles
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        smallest_radius_idx = np.argmin(radii)
        
        # Calculate maximum possible additional expansion while respecting constraints
        # Find the maximum expansion possible for the least constrained circle:
        def max_possible_expansion(current_r, centers, min_dists, n, i):
            # Find the circle with the minimum distance to this circle
            min_dist = min_dists[i]
            # If it's already touching another circle, cannot expand
            if min_dist == 0.0:
                return 0
            # Calculate the available expansion without overlapping
            max_expansion = (min_dist - 2 * current_r) / 2
            return max_expansion if max_expansion > 0 else 0
            
        max_expansion = max_possible_expansion(radii[least_constrained_idx], centers, min_dists, n, least_constrained_idx)
        
        # Apply expansion to least constrained and smallest radius circles
        expansion_factor = max_expansion * 0.8
        expansion_factor_small = expansion_factor * 0.6
        
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor
        new_radii[smallest_radius_idx] += expansion_factor_small
        
        # Ensure no other expansion to prevent invalid configurations
        for i in range(n):
            if i != least_constrained_idx and i != smallest_radius_idx:
                new_radii[i] += 0  # No expansion for other circles
        
        # Create new decision vector with expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10})
    
    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())