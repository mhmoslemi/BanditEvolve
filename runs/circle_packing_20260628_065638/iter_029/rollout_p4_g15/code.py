import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # --- INITIALIZATION PHASE: ENHANCED SPATIAL HASHING AND MULTI-SCALE GEOMETRY ---
    # Initialize with grid pattern but with multi-level spatial hashing to break symmetry
    xs, ys = [], []
    spatial_hashes = np.random.rand(n, 2) * 0.12  # Higher spread for diversity
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add spatial hashing but scaled by radius to maintain feasibility
        # We'll initialize with small radius and then expand
        x = x_center + spatial_hashes[i, 0] * 0.15
        y = y_center + spatial_hashes[i, 1] * 0.15
        # Introduce stagger pattern
        if row % 2 == 1:
            x += 0.5 / cols * 0.75
        xs.append(x)
        ys.append(y)
    
    # Define radius initializer with larger base (for more room for expansion)
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * n
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective function to maximize sum of radii

    # --- CONSTRAINT PHASE: OPTIMIZED FOR NUMERICAL STABILITY AND PARALLEL CHECKING ---
    # Pre-define optimized constraint function for boundaries (direct access to i)
    def boundary_constraints(v, i):
        x, y, r = v[3*i], v[3*i+1], v[3*i+2]
        # Left boundary: x - r >= 0
        left = x - r
        # Right boundary: x + r <= 1
        right = 1.0 - x - r
        # Bottom boundary: y - r >= 0
        bottom = y - r
        # Top boundary: y + r <= 1
        top = 1.0 - y - r
        return np.array([left, right, bottom, top])

    cons = []
    # Pre-define constraints for all circles
    for i in range(n):
        # Bound inequalities: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bound inequalities: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bound inequalities: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Bound inequalities: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints: use vectorized and precomputed indices for speed
    # Precompute all pairs (i,j) as 2D indices, store their indices as tuples
    # For optimization of gradient computation, we'll use a custom function
    # This will ensure proper constraint gradient calculation
    for i in range(n):
        for j in range(i + 1, n):
            # Ensure constraint gradient is correctly derived
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            })

    # --- OPTIMIZATION PHASE: MULTI-STEP ENHANCED OPTIMIZATION WITH STRATEGIC REFINEMENT ---
    # 1st pass: Basic optimization with tighter tolerances
    res1 = minimize(
        neg_sum_radii, v0, method="SLSQP", bounds=bounds,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-9, "eps": 1e-8}
    )
    
    # If we get a success, perform:
    if res1.success:
        # 2nd pass: Add spatial hashing-based perturbation to explore unvisited regions
        # Perturb the centers using spatial hashes but scaled by current radius
        v2 = res1.x.copy()
        radii = v2[2::3]
        hash_perturbation = np.random.rand(n, 2) * 0.03 * np.mean(radii)
        for i in range(n):
            v2[3*i] += hash_perturbation[i, 0]
            v2[3*i+1] += hash_perturbation[i, 1]
        
        # Re-evaluate after perturbation
        res2 = minimize(
            neg_sum_radii, v2, method="SLSQP", bounds=bounds,
            constraints=cons,
            options={"maxiter": 350, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-8}
        )
        
        # 3rd pass: Introduce a global expansion target with constraints
        if res2.success:
            v3 = res2.x.copy()
            # Extract current radii and centers for analysis
            current_radii = v3[2::3]
            # Calculate min distance between all pairs
            centers = np.column_stack([v3[0::3], v3[1::3]])
            dists = np.zeros((n, n))
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx*dx + dy*dy)
            dists = np.maximum(dists, 1e-9)
            # Find the most isolated circle (one with greatest minimal distance to others)
            min_dists = np.min(dists, axis=1)
            isolated_idx = np.argmax(min_dists)

            # Compute expansion budget
            total_curr = np.sum(current_radii)
            target_total = total_curr + 0.0065  # Increase by 0.65% to encourage growth
            max_expansion = (target_total - total_curr) * 1.2  # Add 20% buffer
            min_radius = 1e-4
            max_radius = 0.5
            
            # Create expansion profile: prioritize isolated circle, distribute to others
            expansion_profile = np.zeros(n)
            expansion_profile[isolated_idx] = max_expansion * 0.4  # Assign 40% to isolated
            expansion_profile = expansion_profile / (n) * (target_total - total_curr)
            # Ensure no radius violates bounds
            expansion_profile = np.clip(expansion_profile, -current_radii + min_radius, max_radius - current_radii)
            # Apply expansion with constraints
            expanded_v = v3.copy()
            expanded_v[2::3] += expansion_profile
            
            # Re-evaluate the new configuration with expansion
            res3 = minimize(
                neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                constraints=cons,
                options={"maxiter": 350, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-8}
            )
            
            if res3.success:
                v = res3.x
            else:
                v = res2.x
        else:
            v = res2.x
    else:
        v = res1.x
    
    # Apply final clipping to prevent negative radii
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # Clip to [1e-6, 0.5] to enforce physical validity
    
    return centers, radii, float(radii.sum())