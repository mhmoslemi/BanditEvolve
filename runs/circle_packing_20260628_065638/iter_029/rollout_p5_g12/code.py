import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Use a dynamic initialization pattern with asymmetric geometric hashing for diversity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add spatial perturbation with radius scaling for dynamic geometry
        base_perturb = np.random.uniform(-0.06, 0.06, size=2)
        # Introduce anisotropic asymmetry in x/y directions for non-trivial layouts
        x_perturb = base_perturb[0] * (1.0 + 0.2 * np.sin(np.pi * row))  # row-based x bias
        y_perturb = base_perturb[1] * (1.0 + 0.2 * np.cos(2 * np.pi * col))  # col-based y bias
        x = x_center + x_perturb
        y = y_center + y_perturb
        if row % 3 == 1:  # create a staggered pattern with row-specific offset
            x += 0.4 / cols * (1 if col % 2 == 0 else -1)  # column parity staggering
        xs.append(x)
        ys.append(y)
    
    # Adaptive radius initialization
    base_radius = 0.38 / cols
    radii = base_radius + np.random.uniform(-0.02, 0.02, n)
    # Apply radius asymmetry to break symmetry - higher radii for even-indexed circles
    radii[::2] *= 1.02  # slightly increased even-indexed circles
    # Trim to ensure minimal radii threshold
    radii = np.clip(radii, 1e-5, 0.5)
    v0 = np.zeros(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = radii.copy()

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)]  # tightened for better optimization
    
    # Define objective function with enhanced regularization for convergence
    def neg_sum_radii(v):
        # Primary objective: maximize sum of radii
        radius_sum = np.sum(v[2::3])
        # Secondary regularization: penalize extreme spatial compression to maintain feasibility
        # Use geometric spacing as regularization factor
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Penalize small distances between circles to encourage spacing
        regularization = np.sum(np.maximum(0, (0.03 - dists) * dists))
        # Penalize radius imbalances to encourage uniform distribution
        radius_deviation = np.sum(np.abs(v[2::3] - np.mean(v[2::3])))
        # Weighted sum
        return - (radius_sum - 0.01 * regularization - 0.02 * radius_deviation)

    # Vectorized constraint handling with fixed indices for SLSQP
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraint: use vectorized calculation to optimize
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda capture with fixed indices for consistency
            # Pre-allocate spatial buffers for faster computation in constraints
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})
    
    # First pass optimization: enhanced tolerance and convergence
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-12})
    
    # Asymmetric reconfiguration: spatial rehashing with radius-aware perturbation
    if res.success:
        v = res.x
        # Generate spatial map based on radii distribution
        spatial_map = np.random.rand(n, 2) * 0.05 * (v[2::3] / np.mean(v[2::3]))
        # Apply spatial rehashing: move circles based on their radius influence
        perturbed_v = v.copy()
        for i in range(n):
            # Move by radius-weighted spatial map
            perturbed_v[3*i] += spatial_map[i, 0]
            perturbed_v[3*i+1] += spatial_map[i, 1]
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-12})
    
    # Targeted radius reconfiguration: identify least constrained circle
    if res.success:
        v = res.x
        # Calculate spatial distance matrix with broadcasting
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        # Find minimal distances per circle for constraint analysis
        min_distances = np.min(dists, axis=1)
        # Find the circle with largest minimum distance
        least_constrained_idx = np.argmax(min_distances)
        # Calculate current radius sum
        current_sum = np.sum(v[2::3])
        # Try expanding by max 0.0075 with soft constraints
        target_sum = current_sum + 0.0075
        # Apply targeted expansion with constraint-aware adjustment
        # Create an expansion vector based on spatial influence
        expansion = np.zeros(n)
        expansion[least_constrained_idx] = 0.003  # base expansion
        # Distribute expansion to neighbors within certain distance
        for i in range(n):
            if i != least_constrained_idx:
                dist = np.sqrt((v[3*i] - v[3*least_constrained_idx])**2 + 
                               (v[3*i+1] - v[3*least_constrained_idx+1])**2)
                if dist < 0.1:
                    expansion[i] += 0.001 * (0.1 - dist)
        
        # Create expansion vector with regularization
        expansion = np.clip(expansion, 0.0001, 0.006)
        # Calculate total expansion
        total_expansion = np.sum(expansion)
        # Apply expansion while satisfying constraint violations
        v_new = v.copy()
        v_new[2::3] += expansion
        
        # Validate expansion in a fast mode
        # First validation check
        def fast_validate(v):
            # Validate boundaries
            for i in range(n):
                if v[3*i] < 0 - 1e-12 or v[3*i] + v[3*i+2] > 1 + 1e-12:
                    return False
                if v[3*i+1] < 0 - 1e-12 or v[3*i+1] + v[3*i+2] > 1 + 1e-12:
                    return False
            # Validate overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    if np.sqrt(dx**2 + dy**2) < v[3*i+2] + v[3*j+2] - 1e-12:
                        return False
            return True
        
        if not fast_validate(v_new):
            # Re-evaluate with adjusted expansion
            # Apply linear backtracking in expansion
            while True:
                v_new -= (v_new - v) * 0.1  # reduce expansion
                if fast_validate(v_new):
                    break
        
        # Final optimization with expanded configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11})
    
    # Final processing
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())