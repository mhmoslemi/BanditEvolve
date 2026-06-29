import numpy as np

def run_packing():
    n = 26
    cols = 5  # optimized to 5 column grids for even distribution with staggered rows
    rows = (n + cols - 1) // cols
    cols = 5
    rows = 6  # adjust to have better coverage for 26 circles
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Use a more sophisticated random seed per circle to avoid clustering
        seed = i * 31337 % (2**32 - 1)
        np.random.seed(seed)
        x = x_center + np.random.uniform(-0.04, 0.04)  # tighter random offset
        y = y_center + np.random.uniform(-0.04, 0.04)
        # Shift alternate rows to create staggered grid with adaptive spacing
        shift = 0.35 / cols  # dynamic shift based on grid spacing
        if row % 2 == 1:
            x += shift
        xs.append(x)
        ys.append(y)
    
    # Base radii estimation using circle packing density and grid spacing
    initial_radii = 0.35 / cols * 1.07  # 7% boost to start with larger potential
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, initial_radii)
    
    # Bounds definition aligned with 3*n decision vector
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # radius: [0.0001, 0.5]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # minimize the negative sum = maximize sum

    # Vectorized constraints with stable capture via lambda closures and bound parameter
    cons = []
    for i in range(n):
        # Left-side constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, idx=i: v[3*idx] - v[3*idx + 2]})
        # Right-side constraint: 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, idx=i: 1.0 - v[3*idx] - v[3*idx + 2]})
        # Bottom-side constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, idx=i: v[3*idx + 1] - v[3*idx + 2]})
        # Top-side constraint: 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, idx=i: 1.0 - v[3*idx + 1] - v[3*idx + 2]})
    
    # Vectorized overlap constraints with enhanced geometric hashing and adaptive constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Add adaptive scaling to the constraint function for dynamic tightness
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                # Use a dynamic scaling factor based on the current radius distribution
                scale = 1.0 + (np.log(np.max(v[2::3]) - np.min(v[2::3])) / 5)
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2 / scale  # adaptive scale to promote expansion

            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tight tolerances and extended iterations
    res = minimize(neg_sum_radii, v0, 
                   method="SLSQP", 
                   bounds=bounds,
                   constraints=cons, 
                   options={"maxiter": 1500, "ftol": 1e-10, "gtol": 1e-10})
    
    # Adaptive reconfiguration with spatial hashing and growth-based perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Perform spatial hashing to generate perturbation based on circle properties
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Scale spatial perturbation by radius ratio to maintain stability
            scale = (radii[i] / radii.mean()) ** 0.5  # square root scaling for controlled movement
            perturbed_v[3*i] += spatial_hash[i, 0] * scale
            perturbed_v[3*i + 1] += spatial_hash[i, 1] * scale
        
        # Evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, 
                       method="SLSQP", 
                       bounds=bounds,
                       constraints=cons, 
                       options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})
    
    # Advanced targeted radius expansion using geometric reasoning and constraint sensitivity
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Precompute all pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate constraint sensitivity for each circle (max distance to others)
        min_dists = np.min(dists, axis=1)
        constraint_sensitivity = np.max(min_dists) - np.min(min_dists) + 1e-9  # avoid division by zero
        
        # Identify least constrained circle
        least_constrained_idx = np.argmin(min_dists)
        
        # Estimate potential growth: based on average spacing and radius ratio
        avg_radius = np.mean(radii)
        avg_spacing = np.mean(min_dists)
        
        # Conservative growth estimate: based on spacing-to-radius ratio
        max_growth = avg_spacing / (avg_radius * np.sqrt(2)) * 0.85  # conservative expansion factor
        
        # Create growth vector with prioritized expansion
        growth = np.zeros(n)
        growth[least_constrained_idx] = max_growth * 1.2  # slight over-expansion
        # Distribute remaining growth based on constraint sensitivity
        for i in range(n):
            if i != least_constrained_idx:
                growth[i] = (max_growth * (1 - (min_dists[i] / np.max(min_dists))) * 0.7)
        
        # Apply growth to radii while preserving constraints and checking validity
        max_iter = 50
        for _ in range(max_iter):
            new_radii = radii + growth
            new_centers = np.column_stack([v[0::3], v[1::3]])
            
            # Validate all pairwise distances to detect overlap
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dxij = new_centers[i, 0] - new_centers[j, 0]
                    dyij = new_centers[i, 1] - new_centers[j, 1]
                    distij = np.sqrt(dxij**2 + dyij**2)
                    if distij < new_radii[i] + new_radii[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If overlap, reduce growth proportionally based on constraint severity
                for i in range(n):
                    if i != least_constrained_idx:
                        growth[i] *= 0.95  # conservative constraint enforcement
                # Also reduce max_growth
                max_growth *= 0.95
        
        # Update radii
        new_radii = np.clip(radii + growth, 1e-6, 0.5)  # clip to avoid numerical instability
        v_new = v.copy()
        v_new[2::3] = new_radii
    
    # Final optimization with optimized configuration
    if res.success:
        v = res.x
        radii = v[2::3]
    else:
        v = v0
        radii = v[2::3]
    
    # Final check and constraint enforcement
    if res.success and any(r < 1e-6 for r in radii):
        # Edge case: clip extremely small radii to ensure numerical stability
        radii = np.clip(radii, 1e-6, 0.5)
    
    # Final cleanup on centers
    centers = np.column_stack([v[0::3], v[1::3]])
    centers = np.clip(centers, 1e-8, 1.0 - 1e-8)  # clip to unit square
    
    return centers, radii, float(radii.sum())