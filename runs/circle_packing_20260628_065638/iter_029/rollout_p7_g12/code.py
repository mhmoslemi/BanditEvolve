import numpy as np

def run_packing():
    n = 26
    # Optimize grid distribution and spatial clustering through analytical and probabilistic means
    cols = 5
    rows = (n + cols - 1) // cols
    base_grid = np.zeros((rows, cols))
    
    # Create a 2D grid of base positions with staggered row offset
    for i in range(n):
        row = i // cols
        col = i % cols
        base_grid[row, col] = i
        
    # Construct a more refined grid by introducing controlled clustering and asymmetry
    xs = np.zeros(n)
    ys = np.zeros(n)
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Apply asymmetric spatial displacement based on row parity and circular positioning
        x = x_center + 0.02 * np.sin(2 * np.pi * row / rows)
        y = y_center + 0.02 * np.cos(2 * np.pi * col / cols)
        
        # Apply perturbation for diversity, with increased range to avoid perfect symmetry
        x += np.random.uniform(-0.05, 0.05)
        y += np.random.uniform(-0.05, 0.05)
        
        # Add alternate row shift (staggered grid) to create better inter-circle spacing
        if row % 2 == 1:
            x += 0.5 / cols
        
        xs[i] = x
        ys[i] = y
    
    # Initial radii based on grid spacing with dynamic allocation for better potential
    r0 = 0.32 / cols - 1e-3
    # Add probabilistic radius variation for enhanced optimization space
    r0_var = 0.04 / cols
    radii = r0 + np.random.uniform(-r0_var, r0_var, n)

    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = radii
    
    # Construct bounds with strictness on center positions and radii
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    # Target function: maximize radii sum by minimizing its negative
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Construct constraints using vectorization and more stable lambda capture
    cons = []
    for i in range(n):
        # Left constraint: x_i - r_i > 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right constraint: 1 - x_i - r_i > 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom constraint: y_i - r_i > 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top constraint: 1 - y_i - r_i > 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Construct overlap constraints with more efficient closure capturing and vectorization
    # Use a helper to avoid lambda capture issues
    def get_overlap_constraint(i, j):
        def constraint(v):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
        return constraint
    
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": get_overlap_constraint(i, j)})
    
    # Initial optimization with adaptive control
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-10, "eps": 1e-10})

    # Dynamic reconfiguration strategy: 
    # 1. Analyze interaction pairs to find most constrained circles
    # 2. Rebuild their positions with increased spatial diversity
    # 3. Re-optimize to exploit the new configuration's potential

    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute pairwise distances and identify two most dynamic interacting circles (i,j)
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        
        # Normalize distances for comparison
        dists /= np.max(dists)
        
        # Find two most overlapping interacting circles
        # Strategy: identify pair with max effective overlap (d < r1 + r2 - adjustment)
        effective_overlap = np.zeros(n * n)
        for i in range(n):
            for j in range(n):
                if i < j:  # To avoid double-counting
                    effective_overlap[i * n + j] = (dists[i, j] - (radii[i] + radii[j])) 
        # Select the two most overlapping pairs, first the top two in terms of overlapping
        top_pairs = np.argsort(effective_overlap)[::-1][:2]
        top_i, top_j = divmod(top_pairs[0], n)
        second_i, second_j = divmod(top_pairs[1], n)
        
        # Reconfigure the most active pair using more dynamic spatial hashing
        # This step introduces intentional spatial reordering to break previous constraints
        new_centers = centers.copy()
        # Move these two to opposite corners of the unit square with randomized direction
        corner_shifts = np.random.choice([[-0.1, -0.1], [-0.1, +0.1], [+0.1, -0.1], [+0.1, +0.1]], size=2)
        new_centers[top_i] = [np.random.uniform(-0.1, 1.2), np.random.uniform(-0.1, 1.2)]
        new_centers[top_j] = [np.random.uniform(-0.1, 1.2), np.random.uniform(-0.1, 1.2)]
        # Apply gentle repulsion to other circles to allow expansion
        repulsion_vector = new_centers[top_i] - new_centers[top_j]
        for k in range(n):
            if k != top_i and k != top_j:
                if np.random.random() < 0.1:
                    new_centers[k] -= 0.02 * repulsion_vector  # Apply slight movement away from both
        
        # Transform the new central positions into decision vector
        v_new = v.copy()
        for i in range(n):
            v_new[3*i] = new_centers[i][0]
            v_new[3*i+1] = new_centers[i][1]
        
        # Reevaluate using the modified configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-10, "eps": 1e-10})
        
        # Post-reconfiguration enhancement: 
        # 1. Expand the least constrained circle (i.e. furthest from others)
        # 2. Introduce a novel edge-based adjacency constraint that creates additional
        #    spatial interactions to improve optimization dynamics
        # 3. Apply gradient-based radius expansion with constraint enforcement

        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            
            # Find the least constrained circle (most isolated from others)
            min_distance_to_others = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dx = centers[i, 0] - centers[j, 0]
                        dy = centers[i, 1] - centers[j, 1]
                        min_distance_to_others[i] = np.min([min_distance_to_others[i], np.sqrt(dx**2 + dy**2)])
            least_constrained_idx = np.argmin(min_distance_to_others)
            
            # Find the circle that is furthest from the least constrained one (to expand)
            max_distance_circle_idx = np.argmax(
                np.sqrt((centers - centers[least_constrained_idx, :])**2).sum(axis=1)
            )
            
            # Introduce a new adjacency constraint between these two circles
            # This ensures that they maintain a fixed distance while being expanded
            def adjacency_constraint(v, i=least_constrained_idx, j=max_distance_circle_idx):
                dx = v[3*i] - v[3*j] 
                dy = v[3*i+1] - v[3*j+1] 
                # Define fixed minimal distance constraint (e.g., 0.6 * their average initial radius)
                min_distance = (radii[i] + radii[j]) * 0.8
                return dx*dx + dy*dy - min_distance**2
            
            # Add the new constraint to prevent the two from getting too close (ensuring spatial diversity)
            cons.append({"type": "ineq", "fun": adjacency_constraint})
            
            # Targeted radius expansion on the least constrained circle (not the most isolated)
            # This is a strategic decision to expand the circle that can grow the most without violating space
            expansion_factor = 0.005  # 0.5% boost per iteration
            
            # We will do controlled expansion in stages to avoid constraint violation
            # First, try to expand the circle that can grow the most without violating adjacency
            # We will do a binary search of expansion to find the maximum expansion before conflict
            # Initial expansion in small steps and then larger steps
            
            # First phase: small expansion with high fidelity
            for _ in range(2):  # 2 iterations of small expansion
                # Generate a perturbed version to evaluate growth potential
                perturbed_v = v.copy()
                perturbed_v[3*least_constrained_idx + 2] += expansion_factor
                
                # Revalidate and perform optimization
                res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 200, "ftol": 1e-10, "eps": 1e-10})
                
                if res.success:
                    v = res.x
                else:
                    # If optimization fails, keep previous state
                    break
            
            # Second phase: medium expansion
            for _ in range(2):  # 2 iterations of expansion
                # Generate a perturbed version to evaluate growth potential
                perturbed_v = v.copy()
                perturbed_v[3*least_constrained_idx + 2] += expansion_factor * 1.5
                
                # Revalidate and perform optimization
                res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-10})
                
                if res.success:
                    v = res.x
                else:
                    # If optimization fails, keep previous state
                    break
            
            # Third phase: final expansion
            for _ in range(2):  # 2 iterations of expansion
                # Generate a perturbed version to evaluate growth potential
                perturbed_v = v.copy()
                perturbed_v[3*least_constrained_idx + 2] += expansion_factor * 2.0
                
                # Revalidate and perform optimization
                res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "eps": 1e-10})
                
                if res.success:
                    v = res.x
                else:
                    # If optimization fails, keep previous state
                    break
            
            # Final refinement pass using tight tolerances
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-11})
        
        v = res.x if res.success else v0
    
    # Final cleanup and clipping
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    # Additional check to handle any edge cases due to potential numerical instability
    for i in range(n):
        if centers[i, 0] - radii[i] < -1e-12 or centers[i, 0] + radii[i] > 1 + 1e-12:
            centers[i, 0] = max(min(centers[i, 0], 1.0), 0.0)
        if centers[i, 1] - radii[i] < -1e-12 or centers[i, 1] + radii[i] > 1 + 1e-12:
            centers[i, 1] = max(min(centers[i, 1], 1.0), 0.0)
        # Ensure radius is within bounds
        radii[i] = np.clip(radii[i], 1e-6, 0.5)
    
    return centers, radii, float(radii.sum())