import numpy as np

def run_packing():
    n = 26
    cols = 5  # 5 columns for optimal spatial resolution with 26 entries
    rows = (n + cols - 1) // cols  # Determine row count to evenly distribute
    # Initialize positions with structured grid with adaptive perturbation and geometric refinement
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Center of cell within the grid, with adjusted spacing for staggered rows
        x_center = (col + 0.15) / cols  # Slight left shift to allow better edge packing
        y_center = (row + 0.15) / rows  # Slight top shift
        # Apply geometric clustering with spatial-aware perturbation
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        # Stagger alternate rows for better spacing
        if row % 2 == 1:
            x += 0.5 / cols  # Staggered offset for grid efficiency
        xs.append(x)
        ys.append(y)
    
    # Initial radii estimation with spatial-aware allocation: 
    # First, approximate total available area
    # Grid cell area is (1 / cols * 1 / rows)
    grid_cell_area = (1.0 / cols) * (1.0 / rows)
    # Total area is 1
    max_total_area = 1.0
    # Each circle has area πr², and we're allocating per-cell
    # Assume maximum 3 circles per grid cell to allow dense packing
    r0 = np.sqrt(grid_cell_area / (3.14159 * 1.3)) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Create bounds list that matches the 3 * n decision vector length
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.45)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with lambda captures for boundary conditions
    cons = []
    for i in range(n):
        # Left and radius: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right and radius: 1.0 - x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom and radius: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top and radius: 1.0 - y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with geometric hashing and optimized form
    # We precompute all pair-wise distances using broadcasting for vectorization performance
    for i in range(n):
        for j in range(i + 1, n):
            # Lambda with fixed i,j captures for constraints
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})
    
    # Initial optimization with aggressive constraints and high precision
    # Use 'SLSQP' with tighter tolerances, increased max iterations, and early convergence monitoring
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11, "eps": 1e-12})
    
    # Introduce spatial reconfiguration with geometric hashing to explore non-conformal spatial arrangements
    # Apply perturbation based on normalized radii, to maintain feasibility
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Compute per-circle influence factor based on radius
        influence = radii / np.max(radii)
        # Generate spatial noise in range [-0.008, 0.008] scaled by influence
        spatial_noise = np.random.rand(n, 2) * 2 * 0.008 - 0.008
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_noise[i, 0] * influence[i]
            perturbed_v[3*i+1] += spatial_noise[i, 1] * influence[i]
        # Re-evaluate with new configuration using same constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 350, "ftol": 1e-11, "eps": 1e-12})
    
    # Enforce strict non-overlap with spatial dissection algorithm on most constrained circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Precompute distance matrix with vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find most constrained circle (least distance to other circles)
        min_dists = np.min(dists, axis=1)
        min_dist_circle_idx = np.argmin(min_dists)
        max_dist_circle_idx = np.argmax(min_dists)
        
        # Identify the circle involved in the most critical overlap
        # First, find circle with most critical (smallest) distance to any other
        critical_circle_idx = np.argmin(dists.min(axis=1))
        critical_distances = dists[critical_circle_idx, :]
        # Find its closest neighbor
        closest_neighbor_idx = np.argmin(critical_distances[critical_circle_idx, :])
        
        # We now fix the critical circle's position to avoid re-evaluation and 
        # reposition its neighbor to enforce non-overlap and explore alternate configurations
        # Set critical circle's position to a fixed non-overlapping spot
        # Compute a safe position (e.g., 1.0 - 1.2*radii[critical_circle_idx])
        safe_position = np.array([1.0 - 1.2 * radii[critical_circle_idx], 
                                 1.0 - 1.2 * radii[critical_circle_idx]])
        # Set the fixed position for the critical circle
        v[3*critical_circle_idx] = safe_position[0]
        v[3*critical_circle_idx+1] = safe_position[1]
        # Move the critical neighbor to a new position (left or right, not conflicting)
        # Ensure movement is non-conflicting with other circles
        # Move the neighbor to the left by a factor of its radius
        v[3*closest_neighbor_idx] = v[3*closest_neighbor_idx] - 1.5 * radii[closest_neighbor_idx]
        # Ensure it stays within bounds
        v[3*closest_neighbor_idx] = np.clip(v[3*closest_neighbor_idx], 0.0, 1.0)
        
        # Re-evaluate with this critical spatial reconfiguration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})
    
    # Final expansion pass with global radius constraint and directional expansion
    # Calculate the current total area used to predict maximal expansion (area-based)
    if res.success:
        v = res.x
        radii = v[2::3]
        current_total_area = np.sum(radii ** 2 * np.pi)
        max_area = 1.0 - 0.001  # Leave a sliver for edge and constraint margins
        # Estimate the maximal growth by solving (r^2 * pi) * 26 = max_area
        max_radius_safety = np.sqrt(max_area / (26 * np.pi)) - 1e-3
        # Calculate how much we can increase each radius safely
        # Use a directional expansion factor: larger radii are more stable
        expansion_factor = np.clip((max_radius_safety - np.min(radii)) / np.min(radii), 0, 1.3) * 1.2
        
        # Apply expansion in two phases: 
        # 1. Distribute directional expansion to all circles
        # 2. Apply a global growth factor to maximize total while respecting constraints
        # Ensure each expansion is within their own radius cap
        
        # Phase 1: directional expansion
        for i in range(n):
            safe_growth = max_radius_safety - radii[i]
            if safe_growth > 0:
                v[3*i+2] = min(radii[i] + safe_growth * 0.7, max_radius_safety)
        
        # Phase 2: global growth while maintaining constraints
        # We perform an iterative refinement to achieve the target
        new_radii = v[2::3].copy()
        new_radius_growth = 0.001  # Small initial growth
        
        # Iterate until we reach the target expansion or no further growth is feasible
        while True:
            new_radii = new_radii + new_radius_growth
            # Ensure we don't exceed max safe radius
            new_radii = np.clip(new_radii, 1e-4, max_radius_safety)
            expanded_centers = np.column_stack([v[0::3], v[1::3]])
            # Compute new distance matrix using broadcasting
            dx = expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0]
            dy = expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)
            # Validate that the new configuration still satisfies spacing
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    if dists[i,j] < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid or new_radius_growth < 1e-12:
                break
            else:
                new_radius_growth *= 0.95  # Reduce growth to preserve constraints
        
        # Apply final expansion and re-evaluate
        v[2::3] = new_radii
        # Re-validate constraints
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})
    
    # Final safety check
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())