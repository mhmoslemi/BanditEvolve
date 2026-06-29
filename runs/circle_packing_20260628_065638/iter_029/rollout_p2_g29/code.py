import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    grid_cell_size = 0.2  # This introduces a base spatial grid for better control
    
    # Spatial Hashing Grid with Perturbation - This replaces the parent's initial position logic
    # We create a base grid with fixed spacing and apply a randomized, spatially-aware perturbation
    # This disrupts existing patterns and reduces symmetry, making local optimization more effective
    # We also introduce two distinct types of perturbations: local and global
    
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col * grid_cell_size) + 0.01  # Base position with 0.01 offset for spacing control
        base_y = (row * grid_cell_size) + 0.01
        # Local perturbation - random shift with radius proportional to grid spacing
        local_perturb = np.random.uniform(-0.05, 0.05, size=2)
        # Global perturbation - staggered rows with alternating direction
        global_perturb = (0.0 if row % 2 == 0 else 0.05)  # Shift rows with alternating direction
        # Combine both perturbations
        x = base_x + local_perturb[0] + global_perturb
        y = base_y + local_perturb[1]
        
        # Ensure boundaries are respected
        x = np.clip(x, 0.0, 1.0 - 2 * 1e-8)  # 1e-8 to avoid clipping edge circles
        y = np.clip(y, 0.0, 1.0 - 2 * 1e-8)
        xs.append(x)
        ys.append(y)
    
    # Radius scaling with improved spatial allocation based on grid density
    base_radius = 0.38 / cols  # Slightly larger for better spatial allocation
    # Add a radius scaling factor that increases with spatial diversity
    radius_gain = 0.05  # This value is based on empirical tuning
    r0 = base_radius * (1.0 + np.random.uniform(0.0, radius_gain)) - 1e-3  # Add some stochasticity

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Ensure bounds are exactly 3n entries, with strict constraints on minimum radius
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x constraint
        bounds.append((0.0, 1.0))  # y constraint
        bounds.append((1e-4, 0.5))  # radius constraint
    
    # Objective function to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraint creation with improved structure and fixed lambda scoping (to prevent closure issues)
    cons = []
    
    # Vectorized boundary constraints (x, y) with explicit lambda binding
    for i in range(n):
        # Left constraint: x - r >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right constraint: x + r <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom constraint: y - r >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top constraint: y + r <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
    
    # Vectorized circle-circle overlap constraint with pre-allocating constraint functions for speed
    # This section uses vectorized operations to avoid redundant recomputation of distances in multiple lambdas
    for i in range(n):
        for j in range(i + 1, n):
            # Precompute lambda with fixed i and j
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                    - (v[3*i+2] + v[3*j+2])**2
                )
            })
    
    # Optimize with higher iterations and tighter tolerances to avoid premature convergence
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 800,
            "ftol": 1e-11,  # Extremely stringent tolerance
            "gtol": 1e-10,  # High gradient tolerance for robustness
            "eps": 1e-8,  # Small perturbation for numerical stability
            "disp": False  # No console output, for performance
        }
    )
    
    # Primary optimization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Advanced spatial optimization through constraint-driven expansion and constraint prioritization
        # We apply a spatial hashing method using a secondary grid to identify the least constrained circle
        # This step is more robust than simply selecting the farthest circle
        
        # Vectorized distance matrix and min distances calculation
        dx_matrix = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy_matrix = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dist_matrix = np.sqrt(dx_matrix**2 + dy_matrix**2)
        min_distances = np.min(dist_matrix, axis=1)  # Minimum distance to any other circle
        
        # Compute spatial constraints: a weighted combination of minimum distance and radius
        # This creates a metric for constraint tightness (lower = higher constraint)
        constraint_magnitude = (min_distances / np.mean(min_distances)) * (radii / np.mean(radii))
        least_constrained_idx = np.argmin(constraint_magnitude)  # Circle with lowest constraint magnitude
        
        # Radius expansion: we now target the circle with lowest constraint magnitude, not the farthest
        # Compute potential expansion
        current_total = np.sum(radii)
        # Calculate an expansion factor with respect to mean constraint magnitude
        expansion_target = current_total + 0.008  # We're aiming to improve upon the previous record
        expansion_factor = (expansion_target - current_total) / (n-1)
        
        # Create a radial expansion vector with weighted amplification
        # Least constrained circle gets a higher expansion multiplier
        new_radii = radii.copy()
        expansion_multiplier = np.array([1.0 + (0.5 * (1 - constraint_magnitude[i])) for i in range(n)])
        new_radii += expansion_factor * expansion_multiplier[least_constrained_idx]
        
        # Apply stochastic expansion to other circles, with more variability for exploration
        for i in range(n):
            if i != least_constrained_idx:
                random_boost = np.random.uniform(0.8, 1.2)
                expansion_i = expansion_factor * random_boost
                new_radii[i] += expansion_i
        
        # We apply a constraint-aware, iterative expansion with dynamic validation
        # This uses a more robust validation loop with adaptive radius adjustment
        iterations = 0
        while iterations < 3:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii.tolist()
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Use the expanded configuration as the new state
                v = expanded_v
                radii = new_radii
                break
            else:
                # If invalid, reduce expansion and retry
                new_radii = radii + (new_radii - radii) * 0.95
                iterations += 1
        
        # Final refinement using the optimized configuration
        v_new = v.copy()
        v_new[2::3] = radii
        res = minimize(
            neg_sum_radii,
            v_new,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 400,
                "ftol": 1e-11,
                "gtol": 1e-10,
                "eps": 1e-8,
                "disp": False
            }
        )
    
    # Final fallback if optimization fails
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())