import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Use a hybrid randomization method for centers to avoid symmetry and increase initial diversity
    xs = []
    ys = []
    for i in range(n):
        # Row-major layout with spatial perturbation and staggered rows
        row = i // cols
        col = i % cols
        
        # Base center
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Add spatial noise to break symmetry
        noise = np.random.uniform(-0.15, 0.15, size=2) * (1.0 - (1.0 / rows))
        x = base_x + noise[0]
        y = base_y + noise[1]
        
        # Stagger alternate rows to create a grid-like but less confined arrangement
        if (row + 1) % 2 == 0:
            x += 0.5 / cols
            x = np.clip(x, 0.0, 1.0)
        
        # Add edge-based refinement: shift slightly toward center if near boundary
        if x < 0.1 or x > 0.9:
            x += 0.05 * np.sign(0.5 - x)
        if y < 0.1 or y > 0.9:
            y += 0.05 * np.sign(0.5 - y)
        
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with slightly higher base value to allow more room for optimization
    base_radius = 0.35 / cols - 1e-3
    r0 = base_radius
    # Add some randomness to radii to break symmetry and promote even distribution
    radii_initial = r0 + np.random.uniform(-0.01, 0.01, n)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.clip(radii_initial, 1e-4, 1.0 - 1e-4)
    
    # Define bounds with proper length for decision vector of length 3*n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 1.0 - 1e-4)]
    
    # Objective function to minimize for optimization (maximizer of sum of radii)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized constraints (inequality type) for boundaries
    # Use lambda captures with bound variables to avoid issues with closure capture
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary: 1.0 - (x + r) >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary: 1.0 - (y + r) >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlaps: distance between centers >= sum of radii
    for i in range(n):
        for j in range(i + 1, n):
            # Lambda function using i and j for proper closure binding
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j:
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                        - (v[3*i+2] + v[3*j+2])**2)
            })
    
    # Initial optimization with extended iterations and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-10, "eps": 1e-8})
    
    # If optimization was successful, perform 'shake' heuristic to escape local minima
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Select circles with the smallest radius as candidates for shake
        min_radii_indices = np.argsort(radii)[:3]
        # Generate gentle radial and angular perturbations around these circles
        for idx in min_radii_indices:
            # Add small random perturbation to center
            perturbation_factor = 0.02 + np.random.uniform(0, 0.01)
            radius = radii[idx]
            angle = np.random.uniform(0, 2 * np.pi)
            dx = np.cos(angle) * perturbation_factor * radius
            dy = np.sin(angle) * perturbation_factor * radius
            v[3*idx] += dx
            v[3*idx+1] += dy
            v[3*idx] = np.clip(v[3*idx], 0.0, 1.0)
            v[3*idx+1] = np.clip(v[3*idx+1], 0.0, 1.0)
        
        # Re-optimize with these perturbations
        v = v.clip(0.0, 1.0)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "eps": 1e-8})
    
    # Another layer of refinement with targeted expansion on least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        radii = np.clip(radii, 1e-4, 1.0 - 1e-4)
        
        # Get current centers
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances (vectorized)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute for each circle the minimum distance to others (excluding itself)
        min_dists = np.min(dists, axis=1) - np.diag(dists)  # Exclude diagonal (self)
        min_dists = np.clip(min_dists, 1e-8, np.inf)  # Avoid division by zero
        
        # Find the circle with the largest allowable expansion
        # We use min distance to neighbors as a proxy for flexibility
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute expansion vector with prioritized growth for less constrained circles
        expansion_ratio = 0.5 * np.linspace(1, 1.5, 10)  # Create a range of expansion multipliers
        current_sum = radii.sum()
        target_sum = current_sum + 0.006
        expansion = (target_sum - current_sum) / (n - 1)
        
        # Create new radii vector with targeted expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] = radii[least_constrained_idx] * 1.2  # Boost expansion
        for i in range(n):
            if i != least_constrained_idx:
                max_exp = (target_sum - current_sum) / (n - 1)
                if np.random.rand() < 0.8:
                    new_radii[i] += max_exp * 0.8 * (np.random.rand() + 0.5)
                else:
                    new_radii[i] = np.min([new_radii[i] + max_exp * 0.3, 1.0 - 1e-4])
        
        # Check if new radii would cause overlapping (vectorized check)
        center_array = np.column_stack([v[0::3], v[1::3]])
        max_new_radius = np.max(new_radii)
        is_valid = True
        for i in range(n):
            for j in range(i+1, n):
                dx = center_array[i, 0] - center_array[j, 0]
                dy = center_array[i, 1] - center_array[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                new_dist = dist - (new_radii[i] + new_radii[j])
                if new_dist < -1e-10:
                    is_valid = False
                    break
            if not is_valid:
                break
        
        if is_valid:
            v_new = v.copy()
            v_new[2::3] = new_radii
            # Check if within bounds again
            for i in range(n):
                v_new[3*i] = np.clip(v_new[3*i], 0.0, 1.0)
                v_new[3*i+1] = np.clip(v_new[3*i+1], 0.0, 1.0)
                v_new[3*i+2] = np.clip(v_new[3*i+2], 1e-4, 1.0 - 1e-4)
            v = v_new
        
        # Re-optimize with new radii vector
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "eps": 1e-8})
    
    # Final validation and clipping to ensure all are within bounds
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 1.0 - 1e-6)
    
    return centers, radii, float(radii.sum())