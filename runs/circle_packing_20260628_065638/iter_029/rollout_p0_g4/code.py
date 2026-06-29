import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid with spatial-aware distribution
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid centers calculated with spacing optimized for circle packing
        x_center = ((col + 0.5 + np.random.uniform(-0.02, 0.02)) / cols) * 1.1
        y_center = ((row + 0.5 + np.random.uniform(-0.02, 0.02)) / rows) * 1.1
        # Alternate row staggering with randomized offsets
        if row % 2 == 1:
            x_center += 0.5 / cols * 1.15
        
        # Ensure no center is too close to square edges even with randomized shift
        x_center = np.clip(x_center, 1e-3, 1 - 1e-3)
        y_center = np.clip(y_center, 1e-3, 1 - 1e-3)
        
        # Randomized perturbation with diminishing amplitude based on position 
        x_perturb = np.random.uniform(-0.015, 0.015) * (1 - 0.1 * np.sqrt(np.power(row, 2) + np.power(col, 2)))
        y_perturb = np.random.uniform(-0.015, 0.015) * (1 - 0.1 * np.sqrt(np.power(row, 2) + np.power(col, 2)))
        
        xs.append(x_center + x_perturb)
        ys.append(y_center + y_perturb)
    
    # Initial radii based on grid spacing and adjusted for spatial awareness
    r0 = 0.4 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i
    cons = []
    for i in range(n):
        # Left + radius <= 1 (inequality constraint)
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0 (inequality constraint)
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1 (inequality constraint)
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0 (inequality constraint)
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            # Precompute radius sum as constant for constraint efficiency
            radius_sum = v0[3*i+2] + v0[3*j+2]
            # Constraint function is distance^2 - (r_i + r_j)^2 >= 0
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 ((v[3*i] - v[3*j]) ** 2 + 
                                  (v[3*i+1] - v[3*j+1]) ** 2) - 
                                 (v[3*i+2] + v[3*j+2]) ** 2)})
    
    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2200, "ftol": 1e-11, "eps": 1e-10, "disp": False})
    
    # Spatial disruption layer: Random geometric tiling with dynamic spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate dynamic spatial hashes to break grid-like configurations
        spatial_weights = np.random.rand(2, n) * 0.005
        # Create a new perturbation map that uses weighted spatial awareness and local density
        spatial_perturbation = lambda i: spatial_weights[:, i] * (radii[i] / np.mean(radii) ** 0.8)
        
        # Perturb centers for non-local spatial awareness
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += spatial_perturbation(i)[0]
            new_v[3*i+1] += spatial_perturbation(i)[1]
        
        # Re-optimize with spatial disruption
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 450, "ftol": 1e-11, "eps": 1e-10, "disp": False})
    
    # Constraint prioritization layer: Apply strict boundary enforcement on most constrained circle
    # and global radius expansion constraint
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized pairwise distance matrix with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Constraint matrix to identify overlapping pairs
        overlap_mask = (dists < (radii[:, np.newaxis] + radii[np.newaxis, :]) - 1e-10)
        
        # Calculate local constraint pressure as sum of overlaps for each circle
        constraint_pressure = np.sum(overlap_mask, axis=1)
        most_constrained_idx = np.argmin(constraint_pressure)  # circle with least overlap pressure
        
        # Apply strict boundary enforcement on most constrained circle
        # By creating a dedicated constraint that forces spatial expansion
        def enforce_strict_boundary(v, idx=most_constrained_idx):
            x = v[3*idx]
            y = v[3*idx+1]
            r = v[3*idx+2]
            # We enforce a spatial expansion constraint by using a modified constraint that
            # penalizes proximity to boundaries when the circle has small radius
            return x + r - 0.985 * (1 - (x + r < 0.5) + (x + r > 0.5)) - (x - r)
        
        # Add a new constraint for the most constrained circle
        cons.append({"type": "ineq", 
                     "fun": enforce_strict_boundary})
        
        # Add global radius expansion constraint
        # We encourage a uniform increase in radii while maintaining the non-overlap requirement
        def global_radius_expansion(v):
            # Current total radius
            current_total = np.sum(v[2::3])
            # Calculate a soft expansion factor based on distance to square edges
            # This encourages growth in circles that have more space
            expansion_factor = 0.1 * (1 - (v[3::3] < 0.5) + (v[3::3] > 0.5))
            # Targeting a total radius growth of 0.015
            target_total = current_total + 0.01
            # Encourages growth that matches target_total
            return target_total - current_total + np.sum(expansion_factor) * 0.05
        
        cons.append({"type": "ineq", 
                     "fun": global_radius_expansion})
        
        # Reoptimize with constraint prioritization
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 350, "ftol": 1e-11, "eps": 1e-10, "disp": False})

    # Final configuration refinement with adaptive radius normalization
    if res.success:
        v = res.x
        # Final configuration
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Normalize radii to prevent overly large radii due to constraints
        # While maintaining the non-overlap condition
        if np.sum(radii) > 2.65:  # If sum exceeds the theoretical maximum of 2.65
            # We apply a soft constraint reduction based on position
            # Circles at the corners or edges have their radii reduced
            radii = radii * (1 - 0.02 * np.sqrt((centers[:,0] * (1 - centers[:,0]) + 
                                                 centers[:,1] * (1 - centers[:,1]))))
        
        # Clip radii to prevent negative values
        radii = np.clip(radii, 1e-6, 0.5)
    
    # Output final result
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())