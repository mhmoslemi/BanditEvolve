import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Enhanced initialization: use grid with adaptive jitter and row alternation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Positional base
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Introduce adaptive jitter based on row and column position
        jitter = 0.05 * (1 + (0.15 * (col + row)))
        x_jitter = np.random.uniform(-jitter, jitter)
        y_jitter = np.random.uniform(-jitter, jitter)
        
        # Staggered row offset for better spacing
        row_offset = 0.0 if row % 2 == 0 else 0.5 / cols
        x_center += row_offset
        
        # Apply jitter
        x = x_center + x_jitter
        y = y_center + y_jitter
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.3 + np.random.uniform(-0.1, 0.1)  # Start with a wider initial radius range
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.45)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with explicit i
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with optimized lambda closure
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with explicit i,j and proper scoping
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: np.sqrt(
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                ) - (v[3*i+2] + v[3*j+2])
            })

    # Initial optimization with increased maxiter and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-12, "eps": 1e-10})
    
    # Multi-stage reconfiguration with stochastic spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial influence map for perturbation
        spatial_weights = np.zeros(n)
        for i in range(n):
            spatial_weights[i] = 1.0 - np.min(np.linalg.norm(centers - centers[i], axis=1))
        
        # Adaptive reconfiguration: perturb based on spatial influence
        perturbation_scale = 0.06 + (spatial_weights / np.sum(spatial_weights)) * 0.01
        hash_map = np.random.rand(n, 2) * perturbation_scale[:, np.newaxis]
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_map[i, 0]
            perturbed_v[3*i+1] += hash_map[i, 1]
        
        # Reoptimize with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})
    
    # Targeted radius expansion using gradient-based augmentation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute constraint sensitivity for expansion
        # Use vectorized calculation of minimal distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        
        # Find most under-constrained circle
        constraint_map = 1.0 / (min_dists + 1e-6)
        least_constrained_idx = np.argmax(constraint_map)
        
        # Calculate potential maximal expansion based on spatial constraints
        current_sum = np.sum(radii)
        candidate_radii = radii.copy()
        candidate_radii[least_constrained_idx] *= 1.5  # Safe over-expansion
        
        # Try to expand the least constrained circle while maintaining validity
        expansion_attempts = 0
        expansion_max = 1.5
        while expansion_attempts < 20 and np.sum(candidate_radii) < current_sum + 0.01:
            expansion_attempts += 1
            # Compute minimal distance to others
            for i in range(n):
                if i == least_constrained_idx:
                    continue
                d = np.sqrt( (centers[i,0] - centers[least_constrained_idx,0])**2 + 
                             (centers[i,1] - centers[least_constrained_idx,1])**2 )
                if d < candidate_radii[least_constrained_idx] + candidate_radii[i] - 1e-8:
                    candidate_radii[least_constrained_idx] *= 0.98  # Backtrack
                    expansion_attempts = 0
                    break
            if expansion_attempts >= 20:
                break
        
        # Apply expansion and reoptimize
        new_v = v.copy()
        new_v[2::3] = candidate_radii
        
        # Reoptimize after expansion
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9})
    
    # Final cleanup and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final validation before returning
    valid, error = validate_packing(centers, radii)
    if not valid:
        print(f"Final validation failed: {error}")
        # Fall back to a safe default if invalid
        fallback_centers = np.random.rand(n, 2)
        fallback_centers = np.clip(fallback_centers, 0.05, 0.95)
        fallback_radii = np.full(n, 0.05)
        return fallback_centers, fallback_radii, float(fallback_radii.sum())
    
    return centers, radii, float(radii.sum())