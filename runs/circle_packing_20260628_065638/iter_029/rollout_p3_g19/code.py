import numpy as np

def run_packing():
    n = 26
    
    # --- Step 1: Create spatially optimized initial layout with random perturbation
    # Generate a triangular lattice grid for better packing density
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    xs = np.zeros(n)
    ys = np.zeros(n)
    
    # Generate triangular lattice points with randomized perturbation
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Generate x position with slight perturbation for non-uniform clustering
        base_x = (col + 0.5) / cols
        x = base_x + np.random.uniform(-0.03, 0.03)
        
        # Generate y position with more variation but with alternating row offsets
        if row % 2 == 1:
            base_y = (row + 0.5) / rows
            y = base_y + np.random.uniform(-0.08, 0.08)
        else:
            base_y = (row + 0.5) / rows
            y = base_y + np.random.uniform(-0.06, 0.06)
        
        xs[i] = x
        ys[i] = y
    
    # --- Step 2: Initialize with adaptive radius values
    # Based on local spatial density to optimize expansion potential
    spatial_density = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                spatial_density[i] += 1.0 / (dist + 1e-6)
    
    # Normalize and set initial radii based on spatial density
    r0 = np.sqrt(0.1 / (np.mean(spatial_density) + 1e-6)) * (1.0 / cols)
    r0 = np.clip(r0, 1e-4, 0.4)
    
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

    # --- Step 3: Create bounds with proper constraints
    # All constraints are explicitly tied to their positional indices
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # --- Step 4: Build optimized constraints
    # Use closure-aware constraint functions and vectorization
    cons = []
    
    # Add boundary constraints for all circles
    for i in range(n):
        # Left bound
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right bound
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom bound
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top bound
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
    
    # Add overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate distance squared from i to j
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # --- Step 5: First optimization with adaptive parameters
    # Use SLSQP with adaptive constraints tightening
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # --- Step 6: Targeted perturbation to break symmetry and improve spatial arrangement
    if res.success:
        v = res.x
        
        # --- 6.1: Find the circle with least spatial breathing room for expansion
        # Recompute spatial density based on current positions
        cur_centers = np.column_stack([v[0::3], v[1::3]])
        spatial_density = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = cur_centers[i, 0] - cur_centers[j, 0]
                    dy = cur_centers[i, 1] - cur_centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    spatial_density[i] += 1.0 / (dist + 1e-6)
        
        # Find circle with highest spatial density - most constrained
        most_constrained_idx = np.argmax(spatial_density)
        
        # --- 6.2: Apply multi-layered perturbation to this circle
        # First, perturb in a direction that opens up more space
        direction = np.array([np.random.rand() - 0.5, np.random.rand() - 0.5])
        direction /= np.linalg.norm(direction) + 1e-6
        perturbation = direction * (0.05 + 0.02 * np.power(spatial_density[most_constrained_idx], 0.3))
        
        # Create perturbed configuration
        perturbed_v = v.copy()
        perturbed_v[3*most_constrained_idx] += perturbation[0]
        perturbed_v[3*most_constrained_idx + 1] += perturbation[1]
        
        # Apply second layer of perturbation to maintain balance
        perturbation2 = np.array([0.01 * (np.random.rand() - 0.5), 
                                 0.01 * (np.random.rand() - 0.5)])
        
        perturbed_v[3*most_constrained_idx] += perturbation2[0]
        perturbed_v[3*most_constrained_idx + 1] += perturbation2[1]
        
        # --- 6.3: Optimize with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # --- Step 7: Expand radii based on spatial constraint slack
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute constraint slack for each circle
        constraint_slack = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = dists[i, j]
                    if dist >= 0:
                        constraint_slack[i] += (dist - (radii[i] + radii[j])) * (1.0 / (dist + 1e-6))
        
        # Select circle with maximum constraint slack for targeted expansion
        expansion_circle_idx = np.argmax(constraint_slack)
        if constraint_slack[expansion_circle_idx] > 0:
            # Compute maximum feasible expansion based on minimal constraint
            max_expansion = 0
            for j in range(n):
                if j != expansion_circle_idx:
                    dx = centers[expansion_circle_idx, 0] - centers[j, 0]
                    dy = centers[expansion_circle_idx, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    required = radii[expansion_circle_idx] + radii[j]
                    max_expansion = max(max_expansion, (dist - required) + 1e-6)
            
            # Apply expansion while maintaining constraints
            expansion = max_expansion * 0.5
            expanded_radii = radii.copy()
            expanded_radii[expansion_circle_idx] += expansion
            
            # Build new vector for optimization
            new_v = v.copy()
            new_v[2::3] = expanded_radii
            
            # Re-optimize with expanded configuration
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # --- Step 8: Final configuration and validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())