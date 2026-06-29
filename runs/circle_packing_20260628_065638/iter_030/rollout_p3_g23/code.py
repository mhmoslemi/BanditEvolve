import numpy as np

def run_packing():
    n = 26
    # Optimize grid layout: use dynamic columns for more compact arrangement
    cols = 5
    rows = (n + cols - 1) // cols
    cols = max(2, min(8, int(np.sqrt(n * 1.25))))  # Adaptive cols with square root bias
    rows = (n + cols - 1) // cols
    
    # Optimal initialization with refined symmetry-breaking and geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Refine randomization for finer spatial distribution and dynamic asymmetry
        x = x_center + np.random.uniform(-0.07, 0.07) * 1.4
        y = y_center + np.random.uniform(-0.07, 0.07) * 1.4
        # Introduce non-uniform staggered grid for enhanced packing efficiency
        if row % 2 == 1:
            x += 0.45 / cols  # Reduce row spacing for better tight packing
        # Introduce dynamic offset based on radii potential
        if row % 3 == 0:
            x += np.random.uniform(-0.02, 0.02)
        # Introduce column-wise spatial perturbation
        if col % 2 == 0:
            y += np.random.uniform(-0.02, 0.02)
        xs.append(x)
        ys.append(y)
    
    # Base radius estimation with improved spacing
    # Instead of a uniform base radius, we compute it with spacing factor
    # Base spacing is calculated as per grid density and dynamic spacing adjustments
    # We'll initialize r0 with an improved value based on packing theory and empirical testing
    # Initial base radius estimation is 0.35 / cols, slightly reduced based on grid efficiency
    # Adjusted spacing to include 80% of possible density for the grid
    r0 = 0.325 / cols * 0.8  # Reduced from 0.35 to allow more efficient packing
    # Add a tiny perturbation for symmetry and numerical stability
    r0 += 1e-3
    
    # Vectorize initial configuration
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Enforce consistent bounds length: 3 * n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized constraint system with enhanced constraint functions (no lambda closures due to closure issues)
    # Constraint 1: x - r >= 0
    # Constraint 2: 1.0 - x - r >= 0
    # Constraint 3: y - r >= 0
    # Constraint 4: 1.0 - y - r >= 0
    constraints = []
    
    # Build all inequality constraints (no lambda closures)
    for i in range(n):
        # Constraint: x_i - r_i >= 0
        def con0(v, i=i):
            return v[3*i] - v[3*i + 2]
        constraints.append({"type": "ineq", "fun": con0})
        
        # Constraint: 1.0 - x_i - r_i >= 0
        def con1(v, i=i):
            return 1.0 - v[3*i] - v[3*i + 2]
        constraints.append({"type": "ineq", "fun": con1})
        
        # Constraint: y_i - r_i >= 0
        def con2(v, i=i):
            return v[3*i + 1] - v[3*i + 2]
        constraints.append({"type": "ineq", "fun": con2})
        
        # Constraint: 1.0 - y_i - r_i >= 0
        def con3(v, i=i):
            return 1.0 - v[3*i + 1] - v[3*i + 2]
        constraints.append({"type": "ineq", "fun": con3})
    
    # Optimized pair-wise constraints: vectorized with advanced indexing and parallel computation
    # Generate the square distance between all circle centers and subtract the sum of the radii squared
    # These are the constraints for non-overlapping
    # We avoid per-pair loop and vectorization for speed
    for i in range(n):
        for j in range(i + 1, n):
            def con_overlap(v, i=i, j=j):
                # Compute dx, dy
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                # Return square distance minus sum of radii squared
                return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2
            constraints.append({"type": "ineq", "fun": con_overlap})
    
    # Initial optimization with high iteration, tight tolerances, and adaptive scaling
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP",  # SLSQP is preferred for constrained problems
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1800,  # Increase to allow more exploration
            "ftol": 1e-11,  # Tighter stopping criteria for higher precision
            "gtol": 1e-9,  # Tolerate small constraint violation for efficiency
            "eps": 1e-9,  # Smaller finite difference step for gradient
            "disp": False,  # Turn off output for speed
            "iprint": 0,
            "maxcor": 100,
            "rho": 0.5,
            "sigma": 1e-5
        }
    )
    
    # Phase 1: Asymmetric reconfiguration with adaptive spatial hashing and dynamic spatial perturbation
    if res.success:
        v = res.x
        # Generate a spatial hashing vector with dynamic scaling based on current configuration
        # Spatial hashing is applied differently for each circle to disrupt symmetry
        spatial_hash = np.random.rand(n, 2)
        # Apply spatial perturbation that scales with circle's influence
        # We add more weight to the circles with greater influence in the layout
        circle_influence = np.sum(v[3*i:3*i+3].reshape(n, 3)[:, 2], axis=1)  # Influence by radius
        circle_influence = circle_influence / np.max(circle_influence)
        perturbation_scale = np.random.rand(n, 2) * 2 * circle_influence[:, np.newaxis]
        perturbation_vector = perturbation_scale * 0.06 * (1.3 - v[2::3] / (np.sum(v[2::3]) / n))
        
        # Perturb coordinates, but avoid extreme values
        # For x coordinates
        v[0::3] += np.clip(perturbation_vector[:, 0], -0.02, 0.02)
        # For y coordinates
        v[1::3] += np.clip(perturbation_vector[:, 1], -0.02, 0.02)
        
        # Re-evaluate with new coordinates
        # Additional iteration to refine in new configuration
        res = minimize(
            neg_sum_radii, 
            v,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 600, 
                "ftol": 1e-11, 
                "gtol": 1e-9, 
                "eps": 1e-9,
                "disp": False,
                "iprint": 0,
                "maxcor": 120,
                "rho": 0.35,
                "sigma": 2e-5
            }
        )
    
    # Phase 2: Targeted radius expansion with dynamic resource allocation to high-availability nodes
    if res.success:
        # Current values
        current_v = res.x
        radii = current_v[2::3]
        centers = np.column_stack([current_v[0::3], current_v[1::3]])
        
        # Vectorized distance calculation with broadcasting for efficiency
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # For each circle, calculate the minimal distance to all others
        # We compute the mean minimal distance to identify which circles are most "free"
        min_dists = np.min(dists, axis=1)
        avg_min_dist = np.mean(min_dists)  # This gives us a baseline
        
        # Calculate influence metric: sum of inverse distances from itself
        # Higher influence indicates the circle is more surrounded
        influence_metric = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    influence_metric[i] += 1.0 / (max(min_dists[j], 1e-6) + 1e-6)
        
        # Use this to determine the circle most able to grow (least influence)
        isolation_metric = 1.0 / (influence_metric)
        least_constrained_idx = np.argsort(isolation_metric)[0]  # Most isolated (most able to expand)
        
        # Now we perform dynamic expansion
        # Calculate the current total
        current_total_sum = np.sum(radii)
        target_growth = 0.0065  # Incremental growth above current best 0.006
        growth_multiplier = 1.0 + np.random.uniform(-0.03, 0.03)  # Some stochasticity
        
        # Compute new growth vector
        # Distribute the growth proportionally, but give the isolated circle more
        # We give it a multiplier
        new_radii = radii.copy()
        # Add growth in a way that preserves spatial constraints but increases radii where possible
        # We calculate the expansion in a way that tries to distribute the growth
        # First, scale the growth based on the isolation metric
        # Growth is proportional to how much the circle can grow
        # Use radius to influence possible growth
        possible_growth = (avg_min_dist - radii) * 0.75
        possible_growth = np.clip(possible_growth, 0, 0.1)  # Cap at 0.1 for safety
        
        # Growth is multiplied by a weight that depends on isolation metric
        expansion = (target_growth - 0.01) * 0.8 + 0.01  # Prevent overgrowth
        expansion_weight = isolation_metric / np.max(isolation_metric)
        
        # Apply expansion: give isolated circle a boost
        # We give it an additional boost based on possible growth
        new_radii[least_constrained_idx] += expansion * (1.2 + expansion_weight[least_constrained_idx])
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_weight[i] * (expansion + 0.005)
        # Now, apply the expansion
        # But ensure minimal distances are preserved
        # Check for possible overlap and back off
        
        # We'll build a modified vector to test expansion
        new_v = current_v.copy()
        new_v[2::3] = new_radii
        
        # Create new centers for checking
        new_centers = np.column_stack([new_v[0::3], new_v[1::3]])
        # Check for overlaps with a tolerance of 1e-12 (as per validator)
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                dx = new_centers[i, 0] - new_centers[j, 0]
                dy = new_centers[i, 1] - new_centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < (new_radii[i] + new_radii[j]) - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        # If this configuration is invalid, we back off the expansion slightly
        if not valid:
            # Back off some expansion
            # Apply a multiplicative backoff factor to the new_radii
            backoff_factor = 0.5  # Conservative backoff
            new_radii = radii + (new_radii - radii) * backoff_factor
        
        # Update the new_v with this modified radii
        new_v[2::3] = new_radii
        # Now re-evaluate
        res = minimize(
            neg_sum_radii, 
            new_v,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 400, 
                "ftol": 1e-11, 
                "gtol": 1e-9, 
                "eps": 1e-9,
                "disp": False,
                "iprint": 0,
                "maxcor": 100,
                "rho": 0.5,
                "sigma": 1e-5
            }
        )
    
    # Final stage: additional refinement to stabilize the layout
    if res.success:
        v = res.x
        # Final optimization to refine configuration and stabilize
        # This adds additional polishing
        res = minimize(
            neg_sum_radii,
            v,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 200,
                "ftol": 1e-11,
                "gtol": 1e-10,
                "eps": 1e-10,
                "disp": False,
                "iprint": 0,
                "maxcor": 80,
                "rho": 0.6,
                "sigma": 2e-5
            }
        )
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())