import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Step 1: Enhanced geometric seed initialization with adaptive perturbation and dynamic stagger
    def seed_positions_with_adaptive_stagger_and_perturbation():
        xs = []
        ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            x_center = (col + 0.4) / cols  # Shift toward left for more density
            y_center = (row + 0.4) / rows
            
            # Adaptive perturbation based on row and column for more uniform distribution
            max_perturbation = 0.08 * (1.0 / np.sqrt(n))  # Smaller scale for higher clusters
            row_factor = 0.05 * (1.0 / (rows * cols))  # Adjust based on row/column
            col_factor = 0.04 * (1.0 / (cols * rows))  # Same here
            
            # Perturb based on row, col and their interaction for spatial diversity
            x = x_center + np.random.normal(0, max_perturbation) * (1.0 + 0.01 * row_factor * col_factor)
            y = y_center + np.random.normal(0, max_perturbation) * (1.0 + 0.01 * row_factor * col_factor)
            
            # Row staggered shift with dynamic control based on grid density
            if row % 2 == 1:
                # Use conditional shift based on row density; more aggressive when rows are less dense
                shift_factor = 0.45 / cols  # Reduce shift for denser columns
                if cols > 5:  # If wider, less stagger
                    x += (np.sin(2 * np.pi * row / (cols)) * 0.1) / cols
                else:
                    x += 0.45 / cols
            xs.append(x)
            ys.append(y)
        
        return np.array(xs), np.array(ys)
    
    xs, ys = seed_positions_with_adaptive_stagger_and_perturbation()
    
    # Step 2: Dynamic radius initialization with adaptive spatial constraints and grid-aware scaling
    col_density_weight = 1.0 - (cols - 5) / (cols - 5 + 1) if cols > 5 else 1.0
    row_density_weight = 1.0 - (rows - 5) / (rows - 5 + 1) if rows > 5 else 1.0
    base_radius = (0.34 / cols - 1.5e-4) * col_density_weight * row_density_weight
    
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, base_radius)

    # Step 3: Construct tight bounds ensuring 3n total parameters and 0.5 max radius, but allow 1e-4 min
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    # Step 4: Optimized objective with gradient-aware penalty for better convergence
    def neg_sum_radii_with_gradient_penalty(v):
        # Core objective
        base_sum = -np.sum(v[2::3])
        # Add a penalty for very tight clusters
        cluster_penalty = 0.1 * np.sum(v[2::3] * (v[2::3] > 0.01))
        return base_sum - cluster_penalty

    # Step 5: Vectorized constraints with lambda closures and better closure binding
    # Boundary constraints with improved closure binding
    def create_boundary_constraints(index):
        i = index
        # Left constraint: x - r >= 0
        def left_func(v):
            return v[3*i] - v[3*i+2]
        # Right constraint: x + r <= 1
        def right_func(v):
            return 1.0 - v[3*i] - v[3*i+2]
        # Bottom constraint: y - r >= 0
        def bottom_func(v):
            return v[3*i+1] - v[3*i+2]
        # Top constraint: y + r <= 1
        def top_func(v):
            return 1.0 - v[3*i+1] - v[3*i+2]
        return left_func, right_func, bottom_func, top_func

    # Create boundary constraints for each circle
    cons = []
    for i in range(n):
        left, right, bottom, top = create_boundary_constraints(i)
        cons.append({"type": "ineq", "fun": left})
        cons.append({"type": "ineq", "fun": right})
        cons.append({"type": "ineq", "fun": bottom})
        cons.append({"type": "ineq", "fun": top})
    
    # Step 6: Enhanced pairwise overlap constraint with tighter numerical precision for better convergence
    def create_overlap_constraints(i, j):
        def overlap_func(v):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            return dx**2 + dy**2 - (v[3*i+2] + v[3*j+2])**2
        return {"type": "ineq", "fun": overlap_func}

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append(create_overlap_constraints(i, j))
    
    # Step 7: Initial optimization with adaptive step size and tighter tolerances
    # Initial run with increased iterations and tolerance
    res = minimize(neg_sum_radii_with_gradient_penalty, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "gtol": 1e-10, "eps": 1e-8})
    
    # Step 8: Adaptive configuration refinement with controlled spatial reconfiguration
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute spatial constraints and identify least constrained circles
        # Vectorized pairwise distance calculation via broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute min distances per circle and find the least constrained
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute spatial hashing with adaptive scaling based on circle proximity
        # Create hash with more dynamic perturbation for least constrained
        spatial_hash = np.random.rand(n, 2) * 0.04
        # Apply perturbation proportional to radii and proximity to the least constrained
        # Perturbation is adjusted by 0.2 * (1 - min_dists[least_constrained_idx]/0.5)
        perturbation_scale = 0.1 * (1.0 - min_dists[least_constrained_idx] / 0.5)
        perturbation = spatial_hash * perturbation_scale * radii[np.newaxis, :]
        
        # Apply the spatial perturbation
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += perturbation[i, 0]
            new_v[3*i+1] += perturbation[i, 1]
        
        # Rerun with refined configuration
        res = minimize(neg_sum_radii_with_gradient_penalty, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-11, "gtol": 1e-10, "eps": 1e-6})
        
        # Step 9: Targeted radius expansion of the least constrained circle via soft expansion
        if res.success:
            v_exp = res.x
            centers_exp = np.column_stack([v_exp[0::3], v_exp[1::3]])
            radii_exp = v_exp[2::3]
            
            # Re-check distances for safety and identify least constrained again
            dx_exp = centers_exp[:, np.newaxis, 0] - centers_exp[np.newaxis, :, 0]
            dy_exp = centers_exp[:, np.newaxis, 1] - centers_exp[np.newaxis, :, 1]
            dists_exp = np.sqrt(dx_exp**2 + dy_exp**2)
            min_dists_exp = np.min(dists_exp, axis=1)
            least_constrained_exp_idx = np.argmax(min_dists_exp)
            
            # Calculate current sum and determine expansion
            current_total_sum = radii_exp.sum()
            target_total_sum = current_total_sum + 0.013  # Conservative expansion of +0.013
            expansion_factor = (target_total_sum - current_total_sum) / (n - 1)
            
            # Create expansion vector with targeted expansion on least constrained circle
            new_radii_exp = radii_exp.copy()
            # Apply aggressive expansion to the least constrained circle
            new_radii_exp[least_constrained_exp_idx] += expansion_factor * 1.25  # 25% more than nominal
            # Apply moderate expansion to others with some stochasticity
            for i in range(n):
                if i != least_constrained_exp_idx:
                    # Add some stochastic expansion to break symmetry
                    new_radii_exp[i] += expansion_factor * (0.8 + 0.2 * np.random.rand()) # 80%-120% expansion
            
            # Apply expansion while respecting constraints
            # Check if expansion is possible; if not, reduce expansion
            expanded_v = v_exp.copy()
            expanded_v[2::3] = new_radii_exp
            
            # Constraint validation with 1e-11 tolerance
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_i_j = expanded_v[3*i] - expanded_v[3*j]
                    dy_i_j = expanded_v[3*i+1] - expanded_v[3*j+1]
                    dist_i_j = np.sqrt(dx_i_j**2 + dy_i_j**2)
                    if dist_i_j < expanded_v[3*i+2] + expanded_v[3*j+2] - 1e-11:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                res = minimize(neg_sum_radii_with_gradient_penalty, expanded_v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "gtol": 5e-10, "eps": 5e-8})
            else:
                # If invalid, reduce expansion back to safe bounds
                new_radii_exp = radii_exp.copy()
                for i in range(n):
                    if i != least_constrained_exp_idx:
                        new_radii_exp[i] = np.clip(new_radii_exp[i], 1e-3, 0.5)
                expanded_v = v_exp.copy()
                expanded_v[2::3] = new_radii_exp
                res = minimize(neg_sum_radii_with_gradient_penalty, expanded_v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 5e-10, "eps": 5e-8})
        
        v = res.x
    
    # Final validation and cleanup
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    
    return centers, radii, float(radii.sum())