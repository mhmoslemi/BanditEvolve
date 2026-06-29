import numpy as np

def run_packing():
    n = 26
    cols = 5  # 5-cols grid for more dense and flexible arrangement
    rows = (n + cols - 1) // cols
    
    # Initialize with refined, dynamic grid and adaptive perturbation
    xs = []
    ys = []
    # Use adaptive row offsetting for better spatial utilization
    for i in range(n):
        row = i // cols
        col = i % cols
        # Adaptive centralization based on row height (rows can be uneven)
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Generate perturbation with exponential bias for more extreme positions (row 0 and row -1)
        if row == 0 or row == rows - 1:
            # Push edge rows harder to prevent clustering
            p = np.random.uniform(-0.035, 0.035)
        else:
            p = np.random.uniform(-0.025, 0.025)
        x = x_center + p
        # Sigmoid-based vertical perturbation (less for center rows, higher for edge)
        y_perturbation = 0.025 * np.exp(-2 * np.abs(row - (rows - 1) / 2))
        y = y_center + np.random.uniform(-y_perturbation, y_perturbation)
        
        # Vertical row staggering (alternate rows offset for better packing efficiency)
        if row % 2 == 1:
            y += 0.5 / rows
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius: start at (0.35 / cols) but with adaptive radius growth strategy
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Strict constraint bounds with length 3*n ensuring no mismatch
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries as expected
    
    # Objective function (maximize sum_radii => minimize negative)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized spatial constraints: boundary constraints using lambda with captured i
    # Note: These are now lambda with i as default args properly captured
    cons = []
    for i in range(n):
        # left + radius <= 1 (x - r <= 1 => 1 - x - r >= 0)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2])})
        # right - radius >= 0 (x + r >= 0 => x - r >= 0)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i + 2])})
        # bottom + radius <= 1 (y - r <= 1 => 1 - y - r >= 0)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2])})
        # top - radius >= 0 (y + r >= 0 => y - r >= 0)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i + 1] - v[3*i + 2])})
    
    # Vectorized overlap constraints with careful closure handling
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with explicit closure to fix i and j (critical for optimization)
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i + 1] - v[3*j + 1])**2 - (v[3*i + 2] + v[3*j + 2])**2)
            })
    
    # Initial optimization with increased max iterations and tighter tolerance
    # Initial phase uses basic gradient descent with constraints
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 2000, 
            "ftol": 1e-10, 
            # "eps": 0.0001,  # Reduced epsilon for tight constraint handling
            "disp": False  # Disable verbose to optimize runtime
        })
    
    # First-stage perturbation: reevaluate after initial optimization
    if res.success:
        v = res.x
        # Generate a 'geometric perturbation map' with adaptive magnitude based on radii
        radii = v[2::3]
        # Apply spatial perturbation with radius-dependent strength 
        perturbation_factor = 0.02 * (1.0 + (radii / np.max(radii))) 
        # Random perturbations with direction
        hash_map = np.random.rand(n, 2) * (0.35 / cols)  # Use scaled distribution
        # Generate perturbed vector
        perturbed_v = v.copy()
        for i in range(n):
            # Apply perturbation with radius-dependent scaling
            perturbed_v[3*i] += hash_map[i, 0] * perturbation_factor[i]
            perturbed_v[3*i+1] += hash_map[i, 1] * perturbation_factor[i]
        
        # Re-evaluate with perturbed positions for better local minima escape
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 400,
                "ftol": 1e-11,
                "disp": False
            }
        )
    
    # Advanced optimization stage with multi-objective constraint balancing:
    # 1. Topological regularization to prevent collapse of clusters
    # 2. Radius-driven spatial balancing to spread out circles evenly
    # 3. Gradient-enhanced refinement to converge more precisely
    
    # If previous optimization successful, apply multi-phase refinement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized pairwise distance matrix (avoiding double counting)
        # Using broadcasting for full matrix in O(n^2) time with efficient calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Precompute min distance for each circle for constraint analysis
        min_dists_per_circle = np.min(dists, axis=1)
        min_dist_indices = np.argmin(dists, axis=1)
        
        # Step 1: Dynamic constraint tightening based on cluster density
        # Identify circles with minimal min distance (strongly constrained)
        # Those circles will be prioritized for radius reduction or position adjustment
        
        constrained_indices = np.argsort(min_dists_per_circle)[:5]
        # Create a constraint tightening vector to reduce their radii
        radii_clamped = np.copy(radii)
        radii_clamped[constrained_indices] = np.clip(radii_clamped[constrained_indices], 1e-6, 0.45)
        
        # Step 2: Generate a radial expansion vector with topological constraints
        # Calculate ideal expansion based on average distance to neighbors
        ideal_expansion_per_circle = 0.005  # Base expansion
        expansion_factor_per_circle = ideal_expansion_per_circle * (min_dists_per_circle / np.mean(min_dists_per_circle))
        
        # Adjust expansion factor with inverse proportion to number of near neighbors
        # Avoid circles surrounded by densely packed circles (i.e., low min_dist) from expanding too much
        expansion_factor_per_circle *= (1.0 - (min_dists_per_circle < 0.2 * np.mean(min_dists_per_circle)).astype(float))
        
        # Create a multi-stage expansion vector with dynamic adjustment
        expansion_vector = np.zeros(n)
        # Initialize with base expansion
        expansion_vector += expansion_factor_per_circle * 0.8
        # Apply higher expansion to circles with minimal impact to their neighbors
        for i in range(n):
            if min_dists_per_circle[i] > 0.3 * np.mean(min_dists_per_circle):
                expansion_vector[i] *= 1.3
        
        # Generate a new radius vector with constraint-aware changes
        new_radii = np.copy(radii)
        # Apply expansion only to non-constrained circles
        expansion_mask = (min_dists_per_circle > 0.2 * np.mean(min_dists_per_circle))
        new_radii[expansion_mask] += expansion_vector[expansion_mask]
        new_radii = np.clip(new_radii, 1e-6, 0.45)
        
        # Step 3: Create a new perturbation map with gradient-enhanced spatial adjustment
        # Use the gradient of the distance matrix to identify high-gradient regions
        # These are candidates for slight spatial perturbation to alleviate local constraints
        gradient_dists = np.zeros((n, 2))
        for i in range(n):
            for j in range(n):
                if i != j:
                    di = centers[i, 0] - centers[j, 0]
                    dj = centers[i, 1] - centers[j, 1]
                    # Gradient contribution for circle j at i's location
                    gradient_dists[i, 0] += di * (1.0 / (dists[i, j] + 1e-8))
                    gradient_dists[i, 1] += dj * (1.0 / (dists[i, j] + 1e-8))
        
        # Normalize gradients for perturbation
        gradient_dists -= np.mean(gradient_dists, axis=0)
        gradient_dists /= np.std(gradient_dists, axis=0)
        
        # Generate a more sophisticated perturbation map based on both radius and gradient
        # Perturb positions with higher magnitude for circles with low gradients (less impact on neighbors)
        spatial_perturbation = np.random.rand(n, 2) * (0.025 / (np.mean(min_dists_per_circle) * 0.5))
        spatial_perturbation *= 0.5 * (1.0 - (np.abs(gradient_dists) / np.max(np.abs(gradient_dists))))
        
        # Apply the perturbation to current centers
        perturbed_centers = np.copy(centers)
        perturbed_centers[:, 0] += spatial_perturbation[:, 0]
        perturbed_centers[:, 1] += spatial_perturbation[:, 1]
        
        # Apply edge-bound constraints to perturbed centers
        perturbed_centers = np.vstack([
            np.clip(perturbed_centers[:, 0], 1e-6, 1 - 1e-6),
            np.clip(perturbed_centers[:, 1], 1e-6, 1 - 1e-6)
        ]).T
        
        # Generate a new decision vector with updated positions and radii
        v_new = np.zeros(3 * n)
        v_new[0::3] = perturbed_centers[:, 0]
        v_new[1::3] = perturbed_centers[:, 1]
        v_new[2::3] = new_radii
        
        # Conduct a fine-tuning optimization for precise expansion and spatial adjustment
        res = minimize(
            neg_sum_radii, 
            v_new, 
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 600,
                "ftol": 1e-12,
                "disp": False
            }
        )
    
    # If we have failed to optimize at some stage, revert to a safe baseline
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.45)  # Cap radii to 0.45 to prevent overflow
    
    # Final safety check for radii validity and spatial constraints
    # This is needed due to possible solver drift or invalid constraint satisfaction
    while True:
        # Create a validation pass
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        else:
            # If we're stuck in a non-validated state, fall back to a slightly modified prior state
            # This is a fallback mechanism, not a primary optimization step
            radii = np.clip(radii * 0.99, 1e-6, 0.45)
    
    return centers, radii, float(radii.sum())