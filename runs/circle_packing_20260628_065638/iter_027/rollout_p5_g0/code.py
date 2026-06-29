import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n))) + 1
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a geometric hashing-inspired, randomized staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Base offset for spatial diversity and anti-clustering
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        
        # Apply staggered offset to alternate rows for improved spatial separation
        if row % 2 == 1:
            x += 0.5 / cols * 0.8
        
        # Add noise for perturbation but limit it to avoid collapse near boundaries
        x += np.random.uniform(-0.015, 0.015)
        y += np.random.uniform(-0.015, 0.015)
        
        xs.append(x)
        ys.append(y)
    
    # Start with a reasonable initial radius using area normalization and boundary safety
    r0 = 0.45 / cols - 0.0005  # Slight reduction to prevent overflow in early iterations
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3*n, must match v length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Minimize this to maximize sum of radii

    # Constraint generation using lambda with captured i to avoid closure issues
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints with explicit lambda binding
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with higher tolerance and max iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-8})

    # Apply geometric hashing reconfiguration with adaptive perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate adaptive spatial hashing grid based on current radius distribution
        spatial_hash = np.random.rand(n, 2) * 0.1 * (radii / np.mean(radii))  # radius-scaled perturbation
        perturbed_v = v.copy()
        
        # Apply spatial hashing with bias toward least-constrained circles
        for i in range(n):
            if radii[i] < np.mean(radii):
                perturbed_v[3*i] += spatial_hash[i, 0] * 1.5  # Boost for constrained circles
                perturbed_v[3*i+1] += spatial_hash[i, 1] * 1.5
            else:
                perturbed_v[3*i] += spatial_hash[i, 0] * 0.7  # Conservative for larger circles
                perturbed_v[3*i+1] += spatial_hash[i, 1] * 0.7
        
        # Re-evaluate with new spatial hash
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})

    # Targeted topology reconfiguration using least-constrained circle and adjacency-aware expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation for efficiency
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle (maximize minimum distance to others)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion based on current total and potential capacity
        current_total = np.sum(radii)
        target_total = current_total + 0.008  # 0.008 increase in total sum
        expansion_factor = (target_total - current_total) / (n - 1)  # Per circle expansion
        
        # Introduce directional expansion with spatial hashing and adjacency awareness
        directional_hash = np.random.rand(n, 2) * 0.04  # Slight directional perturbation
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.4  # Slight over-expansion
        
        # Propagate expansion with adjacency-weighted directional influence
        for i in range(n):
            if i != least_constrained_idx:
                # Compute adjacency weight (inverse of distance)
                adj_weight = 1.0 / (np.linalg.norm(centers[least_constrained_idx] - centers[i]) + 1e-12)
                if adj_weight < 0.1:
                    # Boost expansion for tightly packed neighbors
                    expansion = expansion_factor * 1.75
                else:
                    expansion = expansion_factor * 1.25
                # Apply directional expansion with random bias
                new_radii[i] += expansion * (1.0 + directional_hash[i, 0] * 0.3)
        
        # Apply expansion with constraint validation
        max_attempts = 20
        best_v = v.copy()
        best_radii = radii.copy()
        best_valid = False
        
        for attempt in range(max_attempts):
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_v = expanded_v
                best_radii = new_radii
                best_valid = True
                break
            else:
                # If invalid, reduce expansion gradually with stochastic directional bias
                new_radii = radii + (new_radii - radii) * (1.0 - 0.01 * attempt)
        
        if best_valid:
            v = best_v
            radii = best_radii
            centers = np.column_stack([v[0::3], v[1::3]])
        
        # Final re-evaluation with expanded configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())