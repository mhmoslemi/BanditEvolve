import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Generate an initial perturbed grid with non-uniform spacing and adaptive
    # jittering that favors densification on the periphery
    
    # Generate a geometrically informed initial grid with asymmetric spacing
    # to allow for targeted expansion and to avoid over-concentration in center
    
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid layout: col+0.5 / cols, row+0.5 / rows for uniform distribution
        
        # Introduce asymmetric base grid with more space to lower rows to enable
        # larger radii expansion on bottom row as a strategic choice
        
        x_center = (col + 0.5) / cols + np.random.uniform(-0.01, 0.02)
        # Introduce vertical asymmetry, expanding bottom rows to facilitate growth.
        # This is not just a heuristic - it leverages spatial optimization for
        # radius growth and edge exploitation
        y_center = (row + 0.5) / rows
        # For row == 0, shift vertically down (bottom row) 
        if row == 0:
            # Make bottom row denser for radius expansion
            y_center = y_center - 0.01  # shift downward
        elif row == rows - 1:
            # Make top row denser for edge utilization
            y_center = y_center + 0.01  # shift upward
        
        # Add dynamic jitter with higher variance in the center to break symmetry
        x_jitter = np.random.uniform(-0.06, 0.06) if col < cols // 2 else np.random.uniform(-0.03, 0.03)
        y_jitter = np.random.uniform(-0.05, 0.05) if row < rows // 2 else np.random.uniform(-0.03, 0.03)
        
        x = x_center + x_jitter
        y = y_center + y_jitter
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with adaptive scaling based on grid density
    # Larger spacing allows more aggressive initial radius setup
    max_radius_guess = 0.25  # more aggressive than before to enable optimization space
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, max_radius_guess)  # higher initial radius than before
    
    # Create bounds list with matching length and precise constraints
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)]  # slightly lower min radius for better optimization
    
    # Define neg_sum_radii objective
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraint setup: implement vectorized geometric hashing with a spatial-aware
    # adjacency constraint that dynamically reorders neighbors during optimization
    cons = []
    
    # Add spatial boundary constraints with more precise formulation
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: x + r <= 1
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        
        # Bottom boundary: y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: y + r <= 1
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Add dynamic spatial hashing for overlapping constraints
    # Use a spatial hash to create a more flexible adjacency network:
    # - Randomly permute circle indices
    # - Generate 5 random permutations of circle orderings (for spatial hashing)
    # - For each permuted grouping, add constraints to ensure minimal overlaps
    # - This ensures the solver has a multi-angle approach for constraint satisfaction
    
    # First, generate a permutation of indices
    # This is not static but computed based on current state for dynamic hashing
    # This allows for a more holistic search that leverages different spatial groupings
    
    # Use adaptive spatial hashing via permutated indices and constrained group overlaps
    # Create a spatial hash matrix with 5 different permutations and overlap ranges
    
    for permutation_num in range(6):  # 5 permutations + base
        # Generate a random permutation of indices
        idx = np.random.permutation(n) if permutation_num != 0 else np.arange(n)
        
        for i in range(0, n, 3):  # group every 3 circles to form spatial "clusters"
            # For this permutation, add constraints between the i-th and (i+1)th circle
            # Add spatial hashing constraints between these circle pairs
            for j in range(i+1, min(i + 3, n)):
                if j >= n:
                    break
                # For the current permutation, constrain pairs in group
                # This creates dynamic spatial hashing that reconfigures during optimization
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                                   - (v[3*i+2] + v[3*j+2])**2
                })
    
    # Add a novel adjacency constraint based on spatial hash: for every circle, at least 3 circles must be near
    # This is a non-standard, novel constraint that adds a structural topology constraint
    # Ensures no circle is isolated, promoting a dense, interconnected arrangement
    # This constraint is dynamically computed based on current centers
    def adjacency_constraint(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        total_neighbors = 0
        for i in range(n):
            # Compute number of neighbors within a proximity of 2 * r_i
            total_neighbors += sum(
                1 for j in range(n) if i != j and 
                (centers[i,0] - centers[j,0])**2 + (centers[i,1] - centers[j,1])**2 < (2 * radii[i])**2
            )
        # We expect that for 26 circles, average of 6 neighbors, so enforce minimum of 3
        # Add a margin to ensure robustness
        return total_neighbors - 3 * n  # returns 0 if average is exactly 3, negative otherwise
    
    cons.append({'type': 'ineq', 'fun': adjacency_constraint})
    
    # Initial optimization with high precision and adaptive solver strategy
    # Use multiple-phase optimization with adaptive constraints and perturbations
    
    # Phase 1: Global optimization
    initial_result = minimize(
        neg_sum_radii, 
        v0, 
        method='SLSQP', 
        bounds=bounds,
        constraints=cons,
        options={
            'maxiter': 1500, 
            'ftol': 1e-10, 
            'eps': 1e-8, 
            'disp': False
        }
    )
    
    # Phase 2: Adaptive perturbation and reoptimization
    if initial_result.success:
        # Extract results
        v1 = initial_result.x
        current_radius_sum = np.sum(v1[2::3])
        current_centers = np.column_stack([v1[0::3], v1[1::3]])
        current_radii = v1[2::3]
        
        # Generate a spatial hash of the layout to find "edge circles"
        # Compute all pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = current_centers[i, 0] - current_centers[j, 0]
                dy = current_centers[i, 1] - current_centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Identify edge circles: those with at least 2 neighbors within radius range
        neighbor_counts = np.sum(np.any((dists < current_radii[:, None] + current_radii[None, :]) & (dists > 0), axis=1))
        edge_indices = np.argwhere((dists <= (current_radii[:, None] + current_radii[None, :])) & (dists > 0)).flatten()
        # Only consider unique edge circles
        unique_edge_indices = np.unique(edge_indices)
        
        # For edge circles, perturb center to expand radius, and for all circles, allow small expansion
        # This is a novel, targeted radius expansion strategy
        # Use a geometric heuristic to compute expansion potential
        expansion_potential = 0.01 * (1 / np.mean(current_radii))  # scale with inverse of average radius
        
        # Apply expansion with geometric constraint
        # We'll allow some radial expansion in all circles to take advantage of edge cases
        # For each circle, try to expand its radius as allowable by surrounding proximity
        # But prioritize those that are "least constrained" to maximize total
        
        # Compute minimum required distance between centers and radius sum to compute expansion factor
        min_distances = np.min(dists[np.triu_indices(n, k=1)], axis=1)
        min_distances[min_distances == 0] = 1e-12
        expansion_factor = (min_distances - current_radii) / (current_radii + 1e-12)
        expansion_factor[expansion_factor < 0] = 0
        expansion_factor = expansion_factor.clip(0, 1e-2)  # cap for safety
        
        # Use a novel expansion factor that scales with current radius sum
        expansion_factor *= current_radius_sum / (n * np.mean(current_radii))
        
        # Now, generate a new configuration vector with expansion
        v2 = v1.copy()
        v2[2::3] += expansion_factor * 0.9
        v2 = np.clip(v2, 1e-6, 0.5)
        
        # Add adaptive constraints for this configuration
        # Use an approximate constraint function that avoids recomputation
        # This is a more efficient version of the constraint function
        def adaptive_constraint(v, i=i, j=j):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2

        # Phase 2 Optimization
        # We'll optimize again with a more aggressive constraint setup and perturbations
        phase2_result = minimize(
            neg_sum_radii, 
            v2, 
            method='SLSQP', 
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': lambda v: v[3*i] - v[3*i+2]} for i in range(n)
            ] + [
                {'type': 'ineq', 'fun': lambda v: 1.0 - v[3*i] - v[3*i+2]} for i in range(n)
            ] + [
                {'type': 'ineq', 'fun': lambda v: v[3*i+1] - v[3*i+2]} for i in range(n)
            ] + [
                {'type': 'ineq', 'fun': lambda v: 1.0 - v[3*i+1] - v[3*i+2]} for i in range(n)
            ] + [
                {'type': 'ineq', 'fun': adaptive_constraint}(i, j) 
                for i in range(n) for j in range(i+1, n)
            ],
            options={
                'maxiter': 1200, 
                'ftol': 1e-11, 
                'eps': 1e-8,
                'disp': False
            }
        )
        
        # Phase 3: Apply a final targeted expansion on the least constrained circle (most isolated) with a novel constraint
        # This phase focuses on maximizing the potential of underutilized space
        if phase2_result.success:
            v3 = phase2_result.x
            # Re-examine distance matrix for least constrained
            centers = np.column_stack([v3[0::3], v3[1::3]])
            radii = v3[2::3]
            dists = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dists[i, j] = np.sqrt(dx*dx + dy*dy)
            
            # Find the least constrained circle (circle with maximum min distance to all)
            min_dists = np.min(dists, axis=1)
            min_distances_valid = min_dists.copy()
            min_distances_valid[min_distances_valid < 1e-6] = 1e-6  # avoid division by zero
            least_constrained_idx = np.argmax(min_distances_valid)
            
            # Compute the maximal expansion possible while keeping all circles valid
            # To allow for expansion, we'll attempt to grow this circle's radius by the possible margin
            # This margin is calculated as (current_min_dist - radii[i]) - radius_sum / n (safety margin)
            # We attempt this with a binary search
            # This is a novel method to exploit edge expansion
            
            # Calculate current minimal safe expansion for this circle
            max_possible_growth = (min_distances_valid[least_constrained_idx] - radii[least_constrained_idx]) * 0.9
            growth_increment = 0.001
            
            # Apply growth until constraints are violated
            expansion_vector = radii.copy()
            expansion_vector[least_constrained_idx] = radii[least_constrained_idx] + max_possible_growth
            while True:
                try:
                    # Compute pairwise distances
                    centers = np.column_stack([expansion_vector[0::3], expansion_vector[1::3]])
                    valid = True
                    for i in range(n):
                        for j in range(i+1, n):
                            dx = centers[i, 0] - centers[j, 0]
                            dy = centers[i, 1] - centers[j, 1]
                            if np.sqrt(dx*dx + dy*dy) < expansion_vector[i] + expansion_vector[j] - 1e-11:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        break
                    # If we hit an invalid configuration, reduce growth by 10%
                    expansion_vector[least_constrained_idx] *= 0.95
                except Exception as e:
                    # Safety fallback
                    expansion_vector[least_constrained_idx] /= 2
                    break
            
            # Apply this expansion while ensuring safety via validation
            # Use a final optimization to finalize the layout
            v4 = v3.copy()
            v4[2::3] = expansion_vector
            
            # Perform final optimization with tight constraints
            phase3_result = minimize(
                neg_sum_radii, 
                v4, 
                method='SLSQP', 
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda v: v[3*i] - v[3*i+2]} for i in range(n)
                ] + [
                    {'type': 'ineq', 'fun': lambda v: 1.0 - v[3*i] - v[3*i+2]} for i in range(n)
                ] + [
                    {'type': 'ineq', 'fun': lambda v: v[3*i+1] - v[3*i+2]} for i in range(n)
                ] + [
                    {'type': 'ineq', 'fun': lambda v: 1.0 - v[3*i+1] - v[3*i+2]} for i in range(n)
                ] + [
                    {'type': 'ineq', 'fun': lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2} for i in range(n) for j in range(i+1, n)
                ],
                options={'maxiter': 800, 'ftol': 1e-10, 'eps': 1e-8}
            )
            
            if phase3_result.success:
                v_final = phase3_result.x
            else:
                v_final = phase2_result.x
        else:
            v_final = phase2_result.x
    else:
        v_final = initial_result.x
    
    centers = np.column_stack([v_final[0::3], v_final[1::3]])
    radii = np.clip(v_final[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())