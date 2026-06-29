import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))  # Start with a more balanced grid
    grid_size = max(4, cols)  # Ensure grid doesn't collapse to 1x1 due to too few circles
    
    # Adaptive grid: rows and cols determined by spatial constraints
    rows = (n + cols - 1) // cols
    if rows < 2:  # At least two rows to minimize crowding
        rows = 2
        cols = (n + rows - 1) // rows  # Adjust cols to fit new rows
    if cols < 2:  # At least two columns
        cols = 2
        rows = (n + cols - 1) // cols
    
    # Initial distribution with randomized seed for reproducibility and spatial balance
    seed = 42  # Stable seed for consistent optimization landscapes
    np.random.seed(seed)
    
    # Initialize positions with a hybrid structured-random approach
    xs = []
    ys = []
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        # Base grid positions
        x_center = (col_idx + 0.5) / cols
        y_center = (row_idx + 0.5) / rows
        
        # Introduce spatial noise and staggered rows to break symmetry
        row_noise = np.random.uniform(-0.03, 0.03)
        col_noise = np.random.uniform(-0.03, 0.03)
        if row_idx % 2 == 1:  # Stagger alternate rows
            x_center += 0.5 / cols  # Offset alternate rows for non-regular packing
            x_center += np.random.uniform(-0.02, 0.02)
        
        x = x_center + col_noise
        y = y_center + row_noise
        
        # Introduce asymmetric spatial perturbation for edge exploration
        x += 0.04 * (i % 4 - 1.5)  # Create irregular edge spacing
        y += 0.04 * (i % 4 - 1.5)
        
        # Normalize to unit square with a safety margin
        x = np.clip(x, 1e-5, 1.0 - 1e-5)
        y = np.clip(y, 1e-5, 1.0 - 1e-5)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius assignment with density-aware distribution
    initial_density = 0.92 * (n / (cols * rows))
    r0_base = 0.32 / np.sqrt(initial_density) - 1e-3  # More responsive to density
    r0 = np.clip(np.full(n, r0_base), 1e-4, 0.4)  # Safe clipping to prevent numerical issues
    
    # Construct initial vector
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Ensure bounds list and vector match: 3*n elements
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Same length as vector

    # Objective function for optimization: sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Minimize negative sum to maximize sum

    # Generate vectorized constraint functions with correct closure handling (i captured)
    cons = []
    for i in range(n):
        # Boundary constraints (left, right, bottom, top)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraint with vectorized calculation
    # Use broadcasting with fixed indices for better performance
    # Precompute all pairwise comparisons
    for i in range(n):
        for j in range(i + 1, n):
            # Define constraint function with captured i,j to avoid closure issues
            def overlap_cons(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx**2 + dy**2
                return dist_sq - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": overlap_cons})
    
    # First optimization: standard SLSQP with dense constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})

    if res.success:
        v = res.x
        # Post-optimization adaptive refinement phase
        # Step 1: Apply structured geometric hashing to break symmetry
        # For each circle, apply a directional perturbation that preserves grid-like structure
        # This creates a spatial "fingerprint" that helps escape local optima
        hash_factor = 0.045
        for i in range(n):
            # Apply directional perturbation based on spatial grid
            # We use grid coordinates to generate a directional noise
            # Grid: row and column indices
            row_idx = i // cols
            col_idx = i % cols
            direction = np.array([np.sin(np.pi * row_idx / 2), 
                                np.cos(np.pi * col_idx / 2)]) * hash_factor
            # Add direction to positions
            v[3*i] += direction[0]
            v[3*i+1] += direction[1]
            # Scale noise inversely with current radius
            v[3*i] *= (1 + 0.1 * (v[3*i + 2] / r0[i] - 1))
            v[3*i+1] *= (1 + 0.1 * (v[3*i+1 + 2] / r0[i] - 1))
        
        # Second optimization with refined parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    
    if res.success:
        v = res.x
        # Step 2: Targeted refinement of most constrained circles
        # Identify circles with minimal spacing using vectorized distance matrix
        # Efficiently compare all pairs through broadcasting
        
        # Extract centers and radii
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Find circle with minimal minimal distance (most constrained)
        min_distances = np.min(distances, axis=1)
        most_constrained_idx = np.argmin(min_distances)
        
        # Compute radius expansion capability of the most constrained circle
        # This is derived from minimal spacing and potential expansion space
        min_spacing = np.min(distances[most_constrained_idx, :])
        base_radius = min(radii[most_constrained_idx], min_spacing / 2)
        max_possible_radius = np.min(distances[most_constrained_idx, :]) / 2
        allowed_radius_increase = max_possible_radius - base_radius
        
        # Compute expansion factor with a geometric multiplier to explore new states
        expansion_factor = 1.12 * (allowed_radius_increase / base_radius)
        
        # Distribute expansion to all other circles with weighted expansion
        # We bias expansion to circles with more available space
        # This encourages global radius expansion without violating constraints
        # First, compute available expansion space for all circles
        expansion_weights = np.zeros(n)
        for i in range(n):
            if i == most_constrained_idx:
                continue
            min_spacing_i = np.min(distances[i, :])
            base_radius_i = radii[i]
            max_possible_radius_i = min_spacing_i / 2
            allowed_radius_increase_i = max_possible_radius_i - base_radius_i
            expansion_weights[i] = allowed_radius_increase_i / base_radius_i if base_radius_i > 1e-6 else 1.0
        
        # Normalize expansion weights for consistent scaling
        expansion_weights /= np.max(expansion_weights) + 1e-12
        
        # Apply scaled expansion to all circles except the most constrained one
        expansion_amount = expansion_factor * (expansion_weights / np.sum(expansion_weights))
        for i in range(n):
            if i != most_constrained_idx:
                v[3*i + 2] += expansion_amount[i]
        
        # Third optimization with refined radii
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-12})
        
        # Post-optimization validation for the most constrained circle
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            # Explicitly check if the most constrained circle is still valid
            for j in range(n):
                if j != most_constrained_idx:
                    dx = centers[most_constrained_idx, 0] - centers[j, 0]
                    dy = centers[most_constrained_idx, 1] - centers[j, 1]
                    if np.sqrt(dx**2 + dy**2) < radii[most_constrained_idx] + radii[j] - 1e-12:
                        # If still overlapping with others, reduce radius gently
                        v[3*most_constrained_idx + 2] = max(radii[most_constrained_idx] - 1e-4, 1e-4)
                        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                                       constraints=cons, options={"maxiter": 200, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())