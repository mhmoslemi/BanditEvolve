import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Seed for reproducibility (useful for validation, but not strictly needed)
    seed = 42
    np.random.seed(seed)

    def _generate_initial_positions():
        """Create an adaptive staggered grid with randomized initial positions."""
        xs = []
        ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            base_x = (col + 0.5) / cols
            base_y = (row + 0.5) / rows
            # Add jitter to avoid symmetry and increase search space
            x_jitter = np.random.uniform(-0.08, 0.08)
            y_jitter = np.random.uniform(-0.08, 0.08)
            x = base_x + x_jitter
            y = base_y + y_jitter
            # Shift alternate rows to create staggered grid
            if row % 2 == 1:
                x += 0.5 / cols
            xs.append(x)
            ys.append(y)
        return xs, ys

    # Generate the initial positions
    xs, ys = _generate_initial_positions()
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds (exactly 3*n elements)
    bounds = []
    for _ in range(n):
        # x
        bounds.append((0.0, 1.0))
        # y
        bounds.append((0.0, 1.0))
        # radius
        bounds.append((1e-4, 0.5))

    # Objective function (minimize negative sum to maximize radii)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint building: ensure these match 3n-dimensional vector v
    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})

    # Vectorized pairwise overlap constraints: 
    # distance^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            # We apply a more robust and vectorizable closure pattern
            def _make_overlap_func(i, j):
                def f(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                return f
            cons.append({"type": "ineq", "fun": _make_overlap_func(i, j)})

    # Stage 1: Initial optimization with aggressive parameters
    stages = [
        {"method": "SLSQP", "maxiter": 1500, "ftol": 1e-9},
        {"method": "SLSQP", "maxiter": 400, "ftol": 1e-10},
        {"method": "SLSQP", "maxiter": 200, "ftol": 1e-11}
    ]
    
    # Add adaptive perturbation with randomized geometric hashing
    perturbation_scale = [
        (0.05, 0.02),  # stage 0
        (0.02, 0.01),  # stage 1
        (0.01, 0.001)  # stage 2
    ]
    
    # Add dynamic constraint tightening
    constraint_tightening = [
        1.0,
        0.95,
        0.90
    ]
    
    # Initial optimization
    res = minimize(neg_sum_radii, v0, method=stages[0]["method"], bounds=bounds,
                   constraints=cons, options=stages[0])
    if not res.success:
        print(f"Initial optimization failed: {res.message}")
        return (np.column_stack([v0[0::3], v0[1::3]]), 
                np.clip(v0[2::3], 1e-6, None), 
                float(np.sum(np.clip(v0[2::3], 1e-6, None))))

    # Stage 2: Perturb the solution with spatial hashing
    if res.success:
        # Apply spatial hashing: geometric perturbation based on radius
        spatial_hash = np.random.rand(n, 2) * perturbation_scale[1][0]
        # Scale by radius to make the perturbation proportional to size
        spatial_hash *= np.clip(v0[2::3], 1e-4, 0.5) / np.mean(v0[2::3])
        
        perturbed_v = res.x.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        res = minimize(neg_sum_radii, perturbed_v, method=stages[1]["method"],
                       bounds=bounds, constraints=cons, options=stages[1])
        if not res.success:
            print(f"Stage 2 optimization failed: {res.message}")
            return (np.column_stack([res.x[0::3], res.x[1::3]]), 
                    np.clip(res.x[2::3], 1e-6, None), 
                    float(np.sum(np.clip(res.x[2::3], 1e-6, None))))

    # Stage 3: Adaptive radius expansion with constraint tightening
    # We perform dynamic expansion of the circle with minimal interaction
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Calculate pairwise distances once
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # Find the interaction metric: inverse of minimum distance (higher means less confined)
        interaction = np.clip(1.0 / (np.min(dists, axis=1) + 1e-8), 0.0, 10.0)
        # Identify least constrained circle
        least_constrained_idx = np.argmax(interaction)
        
        # Build a list of potential expansion vectors
        expansion_options = []
        # Base expansion based on current total radii
        base_expansion = (0.005 / (2.0)) / (n - 1)  # 50% of 0.005 per circle

        # Create a variety of candidate expansions
        for alpha in [1.2, 1.0, 0.8, 0.5]:
            expansion_candidate = radii.copy()
            expansion_candidate[least_constrained_idx] += base_expansion * alpha
            for i in range(n):
                if i != least_constrained_idx:
                    expansion_candidate[i] += base_expansion * (1.0 + 0.3 * np.random.rand())
            expansion_options.append(expansion_candidate)
        
        best_val = -np.inf
        best_v = v.copy()
        best_rad = radii.copy()
        for exp in expansion_options:
            # Ensure minimal radius is at least 1e-4
            exp = np.clip(exp, 1e-4, 0.5)
            # Create expanded vector
            expanded_v = v.copy()
            expanded_v[2::3] = exp
            # Validate expanded configuration in bulk
            valid = True
            # Precompute distances for all pairs
            all_distances = np.sqrt((centers[:, 0] - centers[:, 0, np.newaxis])**2 + 
                                    (centers[:, 1] - centers[:, 1, np.newaxis])**2)
            for i in range(n):
                for j in range(i+1, n):
                    if all_distances[i, j] < exp[i] + exp[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                current_sum = np.sum(exp)
                if current_sum > best_val:
                    best_val = current_sum
                    best_v = expanded_v
                    best_rad = exp
            
        # Apply the best expansion
        if best_val > 0:
            v = best_v
            radii = best_rad
        else:
            # Fallback: small iterative expansion
            for _ in range(10):
                # Small random expansion
                rad_diff = np.random.uniform(0.0005, 0.001)
                new_radii = radii.copy()
                new_radii[least_constrained_idx] += rad_diff
                # Ensure no circle goes below 1e-4
                new_radii = np.clip(new_radii, 1e-4, 0.5)
                # Check if this new setup is valid
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        if np.sqrt((v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2) < new_radii[i] + new_radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    v[2::3] = new_radii
                    radii = new_radii
                    break
            
        # Re-evaluate with refined configuration
        res = minimize(neg_sum_radii, v, method=stages[2]["method"],
                       bounds=bounds, constraints=cons, options=stages[2])

    # Ensure the solution is valid and fallback to base if needed
    if res.success:
        v = res.x
        # Final validation (safety net)
        if np.isnan(v).any():
            print("NaN detected in final solution - fallback to baseline")
            v = v0
        elif np.any(v < 0):
            print("Negative values detected in final solution - fallback to baseline")
            v = v0
        elif any(v[3*i] - v[3*i+2] < -1e-8 for i in range(n)) or \
            any(1.0 - v[3*i] - v[3*i+2] < -1e-8 for i in range(n)) or \
            any(v[3*i+1] - v[3*i+2] < -1e-8 for i in range(n)) or \
            any(1.0 - v[3*i+1] - v[3*i+2] < -1e-8 for i in range(n)):
            print("Boundary constraint violation - fallback to baseline")
            v = v0
    else:
        # Fallback to base solution
        v = v0
    
    # Final validation before returning
    centers = np.column_stack([v[0::3], v[1::3]])
    radii_arr = v[2::3]
    radii = np.clip(radii_arr, 1e-6, None)
    return centers, radii, float(radii.sum())