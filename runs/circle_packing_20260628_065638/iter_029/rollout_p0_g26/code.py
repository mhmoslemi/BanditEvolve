import numpy as np

def run_packing():
    n = 26
    
    # Step 1: Spatial initialization - use a 5x6 grid with non-uniform distribution to avoid symmetry
    cols = 5
    rows = (n + cols - 1) // cols
    grid_positions = []
    
    # First pass: Generate initial non-symmetrical grid with randomized staggered offset
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        col_offset = np.random.uniform(-0.04, 0.04)
        row_offset = np.random.uniform(-0.02, 0.02)
        
        # Introduce more spacing in lower rows for better expansion potential
        base_x = (col + (0.5 + col_offset)) / cols
        base_y = (row + (0.5 + row_offset)) / rows
        
        # Staggered pattern: offset even and odd rows with different amounts
        if row % 2 == 1:
            base_x += (0.3) / cols
        xs.append(base_x)
        ys.append(base_y)
    
    r0 = 0.28 / cols - 1e-2  # Slightly smaller initial radius for expansion potential
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Step 2: Create bounds with tight constraints for radii (min of 1e-5, max of 0.5)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)]  # 0.00001 min radius to avoid division issues
    
    # Step 3: Define objective function - maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Step 4: Create boundary constraints with lambda closures for efficient and correct capture
    cons = []
    # Left side
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
    # Right side
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
    # Bottom side
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    # Top side
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Step 5: Overlap constraints - use vectorized operations and lambda with proper bound
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                               (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                               - (v[3*i+2] + v[3*j+2])**2})

    # Step 6: Initial optimization with adaptive learning and tighter tolerances
    # Set initial optimizer parameters to balance iteration count and precision
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-12, "eps": 1e-8}
    )
    
    # Step 7: Implement geometric reconfiguration with adaptive spatial perturbation
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric gradient matrix for perturbation - use weighted distance to neighbors
        dist_matrix = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        
        # Weighted gradient matrix: distance to neighbors (inverse relation)
        grad_weights = 1.0 / (dist_matrix + 1e-8)
        grad_weights[range(n), range(n)] = 0  # No self-weight
        grad_weights /= np.sum(grad_weights, axis=1, keepdims=True)
        
        # Generate dynamic perturbation with geometric hashing and gradient weights
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbation_scalar = 0.8 * (radii / np.mean(radii))**0.2
        perturbed_v = v.copy()
        for i in range(n):
            if radii[i] < 1e-4:
                continue
            perturbed_v[3*i] += spatial_hash[i, 0] * perturbation_scalar[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * perturbation_scalar[i]
        
        # Re-evaluate with this adaptive spatial perturbation
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8}
        )
    
    # Step 8: Implement dynamic radius expansion on the least constrained circle via gradient analysis
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distances between all pairs
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        
        # For each circle, min distance to other circles
        min_dists = np.min(dist_matrix, axis=1)
        min_dists = np.clip(min_dists - 1e-8, 1e-6, np.inf)  # Prevent division by zero
        
        # Compute least constrained circle - max distance to neighbors
        least_constrained_idx = np.argmax(min_dists)
        # Get its current radius and neighbors
        r_least = radii[least_constrained_idx]
        centers_least = centers[least_constrained_idx]
        
        # Compute expansion vector targeting this circle
        target_total = np.sum(radii) + 0.0045
        growth_per_circle = (target_total - np.sum(radii)) / n  # Distribute expansion evenly
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += growth_per_circle * 1.5
        
        # Use gradient for final optimization to stabilize convergence
        new_v = v.copy()
        new_v[2::3] = new_radii
        
        res = minimize(
            neg_sum_radii,
            new_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 300, "ftol": 1e-11, "eps": 3e-8}
        )
    
    # Step 9: Final configuration with clipping and validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-5, 0.5)
    
    # Step 10: Final constraint validation and return
    # Validate configuration (should already be done by optimization, but re-check here)
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            if np.sqrt(dx*dx + dy*dy) < radii[i] + radii[j] - 1e-12:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # If invalid, fall back to initial configuration with slight random perturbation
        # To prevent NaNs, add a fallback to base v0 with slight random noise
        new_v = v0.copy()
        noise = np.random.rand(3*n) * 0.0001
        new_v += noise
        centers = np.column_stack([new_v[0::3], new_v[1::3]])
        radii = np.clip(new_v[2::3], 1e-5, 0.5)
        # Ensure no radii are 0, but they should be safe
        for i in range(n):
            if radii[i] < 1e-8:
                radii[i] = 1e-8
        
    return centers, radii, float(radii.sum())