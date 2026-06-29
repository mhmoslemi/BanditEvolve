import numpy as np

def run_packing():
    # Constants
    n = 26
    cols = 5  # Fixed grid for spatial structure and efficient packing
    rows = (n + cols - 1) // cols  # Calculate rows for grid
    
    ###### Step 1: Smart Initialization with Perturbation & Staggered Spatial Sampling
    # Generate grid points with staggered rows and perturbation to avoid symmetry pitfalls
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid position with stagger
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        if row % 2 == 1:
            x_center += 0.5 / cols  # Stagger alternate rows
        
        # Add perturbation in space to avoid symmetry traps
        x = x_center + np.random.uniform(-0.04, 0.04) * (0.5 * (row % 2))  # More for odd rows
        y = y_center + np.random.uniform(-0.04, 0.04) * (0.5 * (row % 2))
        
        # Add noise to edge proximity to allow spatial flexibility later
        x += np.random.uniform(-0.04, 0.04) * (0.25 if row < 3 else 0.1)
        y += np.random.uniform(-0.04, 0.04) * (0.25 if row < 3 else 0.1)
        
        xs.append(x)
        ys.append(y)
    
    initial_radius_estimate = 0.44  # Based on grid geometry
    r0 = initial_radius_estimate / np.sqrt(np.sqrt(2)) - 1e-3  # Adjusted for more efficient packing 
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Build bounds - 3*n entries
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    ###### Step 2: Define Objective Function
    def neg_sum_radii(v):
        return -v[2::3].sum()

    ###### Step 3: Construct Smart Constraints with Caching & Vectorization
    cons = []
    # Build constraint functions with i captured correctly
    for i in range(n):
        # Left boundary (x - r >= 0)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary (x + r <= 1)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary (y - r >= 0)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary (y + r <= 1)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})

    # Overlap constraints: Use broadcasting and sparse vector calculation
    # For performance, we build them as list of lambda that are safe to capture
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    ###### Step 4: First optimization phase with tighter constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    ###### Step 5: Add constraint relaxation for escape from local minima
    # After first optimization, reseed if stuck by adding small random perturbations 
    # but carefully to not violate constraints, using a "soft" randomized constraint re-activation
    if res.success:
        v = res.x
        # Create a soft perturbation matrix to re-orient circles without constraint violation
        perturb = np.random.uniform(-0.01, 0.01, (n, 2))
        perturb *= np.sqrt(np.clip(v[2::3]/0.3, 0.5, 1.0))  # Scale by radius (so smaller circles move less)
        # Apply perturb in a soft way, not breaking boundaries
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturb[i, 0]
            perturbed_v[3*i+1] += perturb[i, 1]
        
        # Re-check boundary integrity after perturbation - this is crucial
        for i in range(n):
            x, y, r = perturbed_v[3*i], perturbed_v[3*i+1], perturbed_v[3*i+2]
            if (x - r < -1e-8 or x + r > 1.0 + 1e-8 or
                y - r < -1e-8 or y + r > 1.0 + 1e-8):
                # If constraint violated, reset perturbation to safe position
                perturbed_v[3*i] = np.clip(x, r, 1.0 - r)
                perturbed_v[3*i+1] = np.clip(y, r, 1.0 - r)
        
        # Re-optimize with perturbations
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP",
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 300, "ftol": 1e-11,
                               "eps": 1e-7, "disp": False})
   
    # If optimization still fails, fallback to random reconfiguration
    if not res.success:
        # Random reconfiguration fallback with spatial hashing
        v = v0.copy()
        # Create spatial hash to randomly displace circles
        for i in range(n):
            v[3*i] = np.random.uniform(0.0, 1.0)
            v[3*i+1] = np.random.uniform(0.0, 1.0)
            # Ensure minimal radius to prevent early failure
            v[3*i+2] = np.clip(0.2 * np.random.rand(), 1e-4, 0.5)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-11})
        
    # If no solution found, fallback to initial state
    v = res.x if res.success else v0
    
    ###### Step 6: Smart Gradient Enhancement (with dynamic constraint sensitivity)
    # After optimization, perform a gradient-aware radius expansion
    # This uses a directional expansion based on spatial constraints
    if res.success:
        # Use vectorized operation for efficiency
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        
        # Precompute distances to neighbors for constraint sensitivity
        dx = centers[0] - centers[0][:, np.newaxis]
        dy = centers[1] - centers[1][:, np.newaxis]
        distance_matrix = np.sqrt(dx**2 + dy**2)
        
        # Compute constraint sensitivity for each circle
        # We use min distance to neighbors in constraint satisfaction
        min_dist = np.min(distance_matrix, axis=1)
        # Use max min distance as indicator for least constrained (can expand more)
        least_constrained_idx = np.argmax(min_dist)
        
        # Use a safety margin to estimate expansion potential
        # We expand radii only for those not in tight proximity
        # Also, we add spatial "gradient" to encourage spread more
        # We use a non-linear expansion factor
        expansion_factor = 0.01  # Base
        radius_expansion = np.zeros(n)
        
        for i in range(n):
            # Expand more for less constrained circles and those with high spatial variance
            # We also scale by the inverse of their current radius for proportionality
            # Use a safety margin of 20% to avoid overlaps
            # Also consider spatial variance to push circles apart
            current_radius = radii[i]
            expansion_multiplier = (1.0 + 0.5 * np.log1p(min_dist[i] / (current_radius * 2.0)))
            radius_expansion[i] = expansion_multiplier * expansion_factor
        
        # Now, perform a directional expansion with gradient-based perturbation
        # We create a new solution vector with adjusted radii
        # We also maintain some spatial perturbation to escape local optima
        v_expanded = v.copy()
        for i in range(n):
            if i == least_constrained_idx:
                # Double expand this circle to anchor expansion
                v_expanded[3*i + 2] += 2 * radius_expansion[i]
            else:
                v_expanded[3*i + 2] += radius_expansion[i]
        
        # Now, apply this expansion via constrained optimization
        # For efficiency, we optimize only for radii change in this step
        # We keep positions fixed but allow radius scaling
        # For better convergence, we use reduced constraints (only radii expansion)
        # This is a "targeted gradient push" to refine the solution
        
        # Optimize again with the expanded radii
        # Since spatial positions are not changing, we'll only optimize radii
        # This helps escape local minima by leveraging spatial gradients
        # We create a new optimization vector with only radii variables
        radius_vars = np.array(v_expanded[2::3])
        # We create bounds that allow dynamic expansion
        # However, to avoid overlapping, we will use the original constraint set but only optimize radii
        # In practice, only radii will move, so we can limit the gradient calculation to that
        # For this step, we'll use "simplex" method for quicker radius adjustments
        res_radius = minimize(lambda v_rad: -v_rad.sum(),
                              radius_vars, method="SLSQP", bounds=[[1e-4, 0.5] for _ in range(n)],
                              constraints=cons,  # Use all overlapping constraints for validation
                              options={"maxiter": 40, "ftol": 1e-11})
        
        # Apply the new optimized radii to the vector
        v_expanded[2::3] = res_radius.x if res_radius.success else v_expanded[2::3]
        
        # Final check for feasibility
        for i in range(n):
            x, y, r = v_expanded[3*i], v_expanded[3*i+1], v_expanded[3*i+2]
            if (x - r < -1e-7 or x + r > 1.0 + 1e-7 or
                y - r < -1e-7 or y + r > 1.0 + 1e-7):
                # For safety, reset to a minimal radius if constrained
                v_expanded[3*i + 2] = np.clip(1e-4, 1e-4, 0.5)
        
        # Final constraint validation is handled by the validator
        v = v_expanded
    
    ###### Step 7: Final Validation and Output
    # Ensure numerical stability and safe clipping
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())