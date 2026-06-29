import numpy as np

def run_packing():
    n = 26
    # Use advanced geometry-driven initialization with hexagonal lattice and adaptive perturbation
    # 5x6 grid for 26 circles: 5 cols, 6 rows with 2 circles in the last row
    cols = 5
    rows = 6
    
    # Initialize with hexagonal lattice with randomized perturbation
    xs = []
    ys = []
    # Start at (0.05, 0.05) to allow more padding around edges
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 0:  # even rows: vertical
                x = (col + 0.25) * 0.76339  # 0.76339 approximates 1 / (cols+0.5)
                y = (row + 0.25) * 0.76339  # same for y
            else:  # odd rows: staggered
                x = (col + 0.25) * 0.76339 + 0.3817  # +0.3817 = 0.5 * 0.76339 for stagger
                y = (row + 0.25) * 0.76339
            # Random perturbation to break symmetry and avoid clustering, scaled by radius
            radius_base = 0.05 * 0.95**row  # geometric decay for smaller circles
            pert = np.random.uniform(-0.03 * radius_base, 0.03 * radius_base, size=2)
            x += pert[0]
            y += pert[1]
            # Ensure x, y are within (0.0, 1.0)
            x_clamped = np.clip(x, 1e-8, 1.0 - 1e-8)
            y_clamped = np.clip(y, 1e-8, 1.0 - 1e-8)
            xs.append(x_clamped)
            ys.append(y_clamped)
    
    # Now check if we have 26 circles
    if len(xs) != n:
        raise RuntimeError(f"Expected 26 initial positions but received {len(xs)}")
    
    # Initialize radii with a decayed base, ensuring that each is unique and non-symmetric
    radii_base = np.array([0.06 * (0.95**(i)) for i in range(n)]).clip(1e-4, 0.5)
    # Add unique perturbation to radii to ensure non-uniformity
    radii_base += 0.005 * np.random.uniform(-0.5, 0.5, size=n).clip(-0.005, 0.005)
    radii_base = np.clip(radii_base, 1e-4, 0.5)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs, dtype=np.float64)
    v0[1::3] = np.array(ys, dtype=np.float64)
    v0[2::3] = radii_base
    
    # Ensure bounds vector has 3*n entries
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # (x, y, r) for each circle

    # Define neg_sum_radii for objective
    def neg_sum_radii(v):
        """Objective function for optimization, minimizing the negative sum of radii."""
        return -np.sum(v[2::3])
    
    # Build constraints (boundary and overlap) with vectorized and lambda-safe setup
    cons = []
    # Boundary constraints for each circle: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    for i in range(n):
        idx = 3 * i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[idx] - v[idx + 2]})  # x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[idx] - v[idx + 2]})  # x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[idx + 1] - v[idx + 2]})  # y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[idx + 1] - v[idx + 2]})  # y + r <= 1
    # Overlap constraints, but with a geometric hashing method to compute only nearby circles (avoid O(N^2))
    def get_neighbor_index_map(v, radius_threshold):
        """Precompute indices where circles are geometrically close (within a threshold distance)."""
        centers = v[0::3], v[1::3]
        center_array = np.column_stack((centers[0], centers[1]))
        dist_matrix = np.sqrt(((center_array[:, None] - center_array[None, :])**2).sum(axis=2))
        neighbor_pairs = np.where(dist_matrix < 1.2 * np.max(radii_base) + 0.01)  # thresholded for performance
        return set(zip(neighbor_pairs[0], neighbor_pairs[1]))
    
    # Use geometric hashing for efficient constraints:
    # For each circle, only check against others that are in its vicinity (based on initial geometry)
    neighbor_indices = get_neighbor_index_map(v0, 0.2)  # Use initial radii to estimate threshold
    # Build constraints only for geometrically relevant pairs
    for i in range(n):
        for j in neighbor_indices:
            if i == j:
                continue
            if j > i:
                idx_i = 3 * i
                idx_j = 3 * j
                # Constraint for overlap: distance**2 >= (r_i + r_j)**2
                cons.append({"type": "ineq", "fun": (lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2) })
    
    # Start optimization with aggressive parameters
    # Initial optimization with SLSQP and tight tolerance
    res = minimize(neg_sum_radii, v0,
                   method="SLSQP", 
                   bounds=bounds,
                   constraints=cons,
                   options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-10, "disp": False})
    
    # Apply enhanced geometric reconfiguration after success, using gradient-based perturbations
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        current_sum_radii = np.sum(radii)
        current_radii_sorted = np.sort(radii)
        
        # Create spatial hash vector with noise that is radius-dependent
        spatial_hash = np.random.rand(n, 2) * 2.0 * np.sqrt(1e-5 + radii) 
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * radii[i] / np.mean(radii)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * radii[i] / np.mean(radii)
        
        # Re-evaluate in new geometry with increased precision and adaptive bounds
        res = minimize(neg_sum_radii, perturbed_v,
                       method="SLSQP", 
                       bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 800, "ftol": 1e-11, "eps": 1e-10, "disp": False})
    
    # Second phase: apply stochastic reconfiguration with soft constraints and geometric expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dists[i, j] = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
        
        # Identify most geometrically isolated circle (max min distance)
        min_distances = np.min(dists, axis=1)
        isolated_idx = np.argmax(min_distances)  # index of largest minimum distance
        isolated_radius = radii[isolated_idx]
        # Compute all other min distances for normalization
        other_min_dists = np.min(dists[np.arange(n)!= isolated_idx, :], axis=1)
        isolated_ratio = 1.5 * (other_min_dists[isolated_idx] / np.mean(other_min_dists))
        # Create expansion factor based on isolation ratio, with geometric adjustment
        expansion_factor = 0.006 * (1 + isolated_ratio * 0.35)  # scale by isolation
        # Expand radius of isolated circle, others shrink slightly to maintain volume
        new_radii = radii.copy()
        new_radii[isolated_idx] += expansion_factor * (1 + np.random.uniform(-0.2, 0.2))  # stochastic expansion
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion_factor * (0.9 + np.random.uniform(-0.1, 0.1)) * (i / np.sum(radii))
        
        # Ensure new_radii within bounds and avoid overlapping
        # Apply soft constraint validation
        while True:
            new_v = v.copy()
            new_v[2::3] = np.clip(new_radii, 1e-4, 0.5)
            new_centers = np.column_stack([new_v[0::3], new_v[1::3]])
            new_radii_array = new_v[2::3]
            
            # Soft check for overlap in a geometrically efficient way
            soft_overlap = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = new_centers[i, 0] - new_centers[j, 0]
                    dy = new_centers[i, 1] - new_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii_array[i] + new_radii_array[j] - 1e-12:
                        soft_overlap = True
                        break
                if soft_overlap:
                    break
            
            if not soft_overlap:
                break
            # If overlapping, reduce expansion slightly
            expansion_factor *= 0.98
            new_radii = new_radii - expansion_factor * (new_radii - radii)
    
        # Apply modified radii and optimize again
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new,
                       method="SLSQP", 
                       bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 800, "ftol": 1e-11, "eps": 1e-10, "disp": False})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())