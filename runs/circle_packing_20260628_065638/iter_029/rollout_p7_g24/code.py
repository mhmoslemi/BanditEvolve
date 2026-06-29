import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Initialize with refined grid staggering and spatial distribution
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Start with a base grid but add spatially asymmetric jitter to escape local optima
        x_center = (col + 0.5) / cols + np.random.uniform(-0.08, 0.02)
        y_center = (row + 0.5) / rows + np.random.uniform(-0.02, 0.08)
        # Alternate row staggering with adaptive spacing
        if row % 2 == 1:
            x_center += 0.5 / cols * (1.0 if np.random.rand() < 0.3 else -0.8)
        xs.append(x_center)
        ys.append(y_center)
    
    # Initialize radii with more aggressive starting point for convergence
    r0 = 0.4 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.35)]  # Reduced upper bound to avoid edge clashes
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized constraint definitions with lambda closures that safely capture i
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with spatial hashing for more efficient calculation
    # Use pre-allocated distance grid and matrix multiplication for faster pairwise distance check
    # Create a sparse distance grid to avoid redundant computation
    for i in range(n):
        for j in range(i + 1, n):
            # Use nested lambda to safely bind i and j
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2
            })
    
    # First pass: global optimization under soft constraints
    # We'll use a hybrid approach with initial optimization + spatial reconfiguration
    res = minimize(neg_sum_radii, v0, method="SLSQP", 
                   bounds=bounds, constraints=cons, 
                   options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-9})
    
    # If not converged, try alternative configuration based on spatial hash map
    if not res.success:
        res = minimize(neg_sum_radii, v0, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 1500, "ftol": 1e-11, "gtol": 1e-10})
    
    if res.success:
        v = res.x
        # Compute spatial relationships once and store
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Find 2 most dynamically interacting circles: use pairwise distance vector
        dists = np.zeros(n * n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i * n + j] = np.sqrt(dx**2 + dy**2)
                dists[j * n + i] = dists[i * n + j]
        # Find top two interacting pairs by sorting distance matrix
        idxs_sorted = np.argsort(dists)
        # Find the top 2 unique circle pairs
        interacting_pairs = []
        seen = set()
        for idx in idxs_sorted:
            i = idx // n
            j = idx % n
            if i < j and (i, j) not in seen:
                interacting_pairs.append((i, j))
                seen.add((i, j))
                if len(interacting_pairs) == 2:
                    break
        # Isolate these 2 circles for reconfiguration
        isolated_indices = set()
        for i, j in interacting_pairs:
            isolated_indices.add(i)
            isolated_indices.add(j)
        # Create a copy for spatial reconfiguration
        reconfigured_v = v.copy()
        # Compute a spatial hash map with higher perturbation for reconfiguration
        # Scale perturbation by radii for more dynamic adjustment
        hash_map = np.random.rand(n, 2) * (0.01 + 0.01 * (radii / np.mean(radii)))
        for i in isolated_indices:
            reconfigured_v[3*i] += hash_map[i, 0]
            reconfigured_v[3*i+1] += hash_map[i, 1]
        # Re-evaluate with reconfigured state
        res = minimize(neg_sum_radii, reconfigured_v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-10})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Find the least constrained circle: optimize for maximum minimal distance
        dists = np.zeros(n * n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i * n + j] = np.sqrt(dx**2 + dy**2)
                dists[j * n + i] = dists[i * n + j]
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        # Ensure the least constrained circle is not one of the 2 interacting ones
        while least_constrained_idx in isolated_indices:
            least_constrained_idx = np.argsort(min_dists)[-2]  # pick next best
        
        # Apply targeted spatial perturbation to the most dynamic interaction
        # Use exponential scale based on radii to avoid large displacement
        scale = 0.02 + 0.01 * (radii[least_constrained_idx] / np.mean(radii))
        dx = np.random.normal(0, scale)
        dy = np.random.normal(0, scale)
        v[3*least_constrained_idx] += dx
        v[3*least_constrained_idx+1] += dy
        # Re-evaluate with perturbed spatial configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-10})
    
    if res.success:
        v = res.x
        # Final refinement: apply a geometric hashing-based expansion to the least constrained
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.zeros(n * n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i * n + j] = np.sqrt(dx**2 + dy**2)
                dists[j * n + i] = dists[i * n + j]
        # Compute final constraint matrix
        # Targeted expansion vector with controlled expansion on least constrained
        new_radii = radii.copy()
        # Calculate expansion by proportional distance from the least constrained
        # Create a growth mask that increases expansion for circles with more "room"
        min_dist_from_center = np.min(np.sqrt((centers - [0.5, 0.5])**2), axis=1)
        # Use a spatial hashing factor to scale expansion based on distance from center
        hash_map = np.random.rand(n, 2) * (0.005 + 0.005 * (min_dist_from_center / np.mean(min_dist_from_center)))
        # Generate target vector with controlled growth
        target_radii = new_radii.copy()
        # Add a small controlled growth to the least constrained
        target_radii[least_constrained_idx] += max(0.002 - 0.001 * np.random.rand(), 0.001)
        # Apply spatial hashing-based expansion across all
        for i in range(n):
            # Add a small perturbation based on radius and position
            new_rad = new_radii[i] + max(0.0005 - 0.0002 * np.random.rand(), 0.0003) * (1.0 + hash_map[i, 0])
            target_radii[i] = np.clip(new_rad, 1e-6, 0.35)  # Clamp to safe radius limit
        
        # Apply this expansion vector and validate
        while True:
            v_expanded = v.copy()
            v_expanded[2::3] = target_radii
            # Recompile centers to validate
            expanded_centers = np.column_stack([v_expanded[0::3], v_expanded[1::3]])
            # Validate against all pairs
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < target_radii[i] + target_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # If invalid, scale back expansion
                target_radii = new_radii + (target_radii - new_radii) * 0.85

        # Final optimization with refined spatial configuration
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 200, "ftol": 1e-11, "gtol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.35)  # Final clamp for safety
    return centers, radii, float(radii.sum())