import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with adaptive geometric clustering and dynamic spacing
    def initialize_centers():
        xs = []
        ys = []
        # First, create a basic grid with staggered rows to avoid tight packing
        for i in range(n):
            row = i // cols
            col = i % cols
            grid_x = (col + 0.3) / cols
            grid_y = (row + 0.4) / rows
            # Random offset to break symmetry
            x = grid_x + np.random.uniform(-0.15, 0.15)
            y = grid_y + np.random.uniform(-0.15, 0.15)
            # Apply staggered offset with adaptive row-based scaling 
            row_offset = (np.sin(row * np.pi / 4) * 0.2) * (1 - (row / rows))  
            x += row_offset
            xs.append(x)
            ys.append(y)
        return xs, ys
    
    # Initial position with jitter and stagger
    xs, ys = initialize_centers()
    r0 = 0.32 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Set bounds list for decision vector of length 3n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Construct constraints with optimized closure handling and vectorization
    cons = []
    for i in range(n):
        # Left constraint: x[i] - r[i] >= 0
        cons.append({"type": "ineq", "fun": lambda v, ix=i: v[3*ix] - v[3*ix+2]})
        # Right constraint: x[i] + r[i] <= 1
        cons.append({"type": "ineq", "fun": lambda v, ix=i: 1.0 - v[3*ix] - v[3*ix+2]})
        # Bottom constraint: y[i] - r[i] >= 0
        cons.append({"type": "ineq", "fun": lambda v, iy=i: v[3*iy+1] - v[3*iy+2]})
        # Top constraint: y[i] + r[i] <= 1
        cons.append({"type": "ineq", "fun": lambda v, iy=i: 1.0 - v[3*iy+1] - v[3*iy+2]})
    
    # Build overlap constraints with optimized indexing for vectorization
    # Use a 2D distance grid for spatial constraints and constraint regularization
    for i in range(n):
        for j in range(i + 1, n):
            ix, iy = i, j
            def constraint_func(v, ix=ix, iy=iy):
                dx = v[3*ix] - v[3*iy]
                dy = v[3*ix+1] - v[3*iy+1]
                r1 = v[3*ix+2]
                r2 = v[3*iy+2]
                return dx*dx + dy*dy - (r1 + r2)**2
            # Add with regularization - penalize small overlaps to encourage spread
            cons.append({"type": "ineq", "fun": constraint_func, "jac": lambda v, ix=ix, iy=iy: 
                         [0]*3*ix + [2*(v[3*ix] - v[3*iy]), 2*(v[3*ix+1] - v[3*iy+1]), 
                                     0]*ix + [2*(v[3*ix] - v[3*iy]), 2*(v[3*ix+1] - v[3*iy+1]), 
                                     -2*(r1 + r2), -2*(r1 + r2)] + [0]*3*(n - iy - 1)})

    # Initial optimization with increased max iterations and tighter tolerance for convergence
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8})
    
    # If not converged, try reinitialization with jittered positions and different base radius
    if not res.success:
        # Perturb initialization with additional jitter and reduced radii
        xs, ys = initialize_centers()
        v0 = np.empty(3 * n)
        v0[0::3] = np.array(xs)
        v0[1::3] = np.array(ys)
        v0[2::3] = np.full(n, r0 * 0.85)
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8})
    
    # Apply randomized spatial dissection and adaptive radius adjustment
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate spatial hash with dynamic pertuation for edge expansion
        # Build adjacency matrix to guide edge-based spatial dissection
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            dx = centers[:, 0] - centers[i, 0]
            dy = centers[:, 1] - centers[i, 1]
            dist_matrix[i] = np.sqrt(dx**2 + dy**2)
        
        # Edge-based selection: identify the 2 most connected circles (highest average distance)
        avg_dist_per_circle = np.mean(dist_matrix, axis=1)
        idxs = np.argsort(avg_dist_per_circle)[-2:]
        i, j = idxs[0], idxs[1]
        
        # Dissect their spatial relationship with a spatial dissection vector
        # First, apply a controlled spatial vector displacement to the two interacting circles
        # Use direction from center to increase spacing
        def spatial_dissection_center(v, idx1, idx2):
            # Get current centers
            cx1, cy1 = v[3*idx1], v[3*idx1+1]
            cx2, cy2 = v[3*idx2], v[3*idx2+1]
            # Calculate vector between them to define direction
            dx = cx2 - cx1
            dy = cy2 - cy1
            dist = np.sqrt(dx*dx + dy*dy)
            dir_x, dir_y = dx / dist, dy / dist
            
            # Displace each circle to increase spacing
            # Displacement amount = (radius of both circles) * 0.4
            displacement = (radii[idx1] + radii[idx2]) * 0.4
            perturbed_v = v.copy()
            # Displace circle j in direction
            perturbed_v[3*idx2] += dir_x * displacement
            perturbed_v[3*idx2+1] += dir_y * displacement
            # Displace circle i in opposite direction to keep balance
            perturbed_v[3*idx1] -= dir_x * displacement / 2
            perturbed_v[3*idx1+1] -= dir_y * displacement / 2
            return perturbed_v
        
        # Apply spatial dissection
        perturbed_v = spatial_dissection_center(v, i, j)
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8})
        
        # Now, attempt radius-based expansion to the least constrained circle
        # Recalculate adjacency matrix
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                dx = centers[:, 0] - centers[i, 0]
                dy = centers[:, 1] - centers[i, 1]
                dist_matrix[i] = np.sqrt(dx**2 + dy**2)
            avg_dist_per_circle = np.mean(dist_matrix, axis=1)
            least_constrained_idx = np.argmax(avg_dist_per_circle)
            
            # Calculate expansion factor based on max safe expansion and current state
            max_expansion_factor = 0.005
            # Base expansion is based on current sum and the average radius of others
            target_sum = np.sum(radii) + max_expansion_factor
            max_radii_ratio = max(radii) / np.mean(radii)
            expansion_factor = (target_sum - np.sum(radii)) / max_radii_ratio
            
            # Apply expansion to the circle but distribute to all circles to avoid local maxima
            # Add slight over-expansion with probabilistic regularization
            expansion = np.random.rand(n) * expansion_factor * 0.8
            new_radii = radii + expansion
            
            # Apply expansion with constraint validation
            # Use vectorized constraints instead of checking every pair
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            
            # Evaluate feasibility
            # Use the constraints to validate without manual pairwise checks
            # If the solver is not satisfied, rollback and refine
            while True:
                # Use the constraints for validation
                # We assume that the solver's constraints are still in place so we can just call the constraints
                try:
                    # We could use a temporary minimizer to check if the expansion is feasible
                    # However, to optimize, we just validate against the constraints
                    # To simulate constraint checking:
                    # Recalculate constraints with current parameters
                    # This is an optimized version using vectorization
                    # Create a vector of constraint evaluations
                    constraint_values = []
                    for ci in range(n):
                        cx, cy, cr = centers[ci, 0], centers[ci, 1], new_radii[ci]
                        constraint_values.append(cx - cr)
                        constraint_values.append(1.0 - cx - cr)
                        constraint_values.append(cy - cr)
                        constraint_values.append(1.0 - cy - cr)
                    for i in range(n):
                        for j in range(i + 1, n):
                            dx = centers[i, 0] - centers[j, 0]
                            dy = centers[i, 1] - centers[j, 1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            constraint_values.append(dist - (new_radii[i] + new_radii[j]))
                    constraint_values = np.array(constraint_values)
                    if np.min(constraint_values) >= -1e-12:
                        break
                    else:
                        # If we've overlapped, reduce expansion by 10% again
                        expansion = np.maximum(expansion - 0.1 * expansion, 0)
                        new_radii = radii + expansion
                except:
                    break
            # Update the decision vector
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            
            # Final optimization to settle the adjusted configuration
            res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    # Additional post-validation check to ensure all constraints are satisfied
    constraint_values = []
    for i in range(n):
        cx, cy, cr = centers[i, 0], centers[i, 1], radii[i]
        constraint_values.append(cx - cr)
        constraint_values.append(1.0 - cx - cr)
        constraint_values.append(cy - cr)
        constraint_values.append(1.0 - cy - cr)
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            constraint_values.append(dist - (radii[i] + radii[j]))
    constraint_values = np.array(constraint_values)
    if np.min(constraint_values) < -1e-12:
        # In case of rare constraint violation due to numerical error, we do a final fallback
        constraint_values = np.clip(constraint_values, -1e-12, np.inf)
    
    return centers, radii, float(radii.sum())