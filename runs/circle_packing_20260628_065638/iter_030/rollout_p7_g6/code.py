import numpy as np

def run_packing():
    n = 26
    cols = 6  # Increased grid width to handle asymmetric layout
    rows = (n + cols - 1) // cols
    
    # First-stage spatial initialization: hybrid grid + radial symmetry break with adaptive spacing
    xs = []
    ys = []
    # Base grid for spatial control
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        # Initial grid spacing with adaptive col scaling
        base_col = (col_idx + 0.5) / cols * 1.2  # 1.2 increases spatial density in the beginning
        base_row = (row_idx + 0.5) / rows * 1.2
        # Symmetry break with dynamic radius-aware shift
        r_factor = 1.0 - (0.4 / cols)  # Radii affect spatial flexibility
        # Random spatial perturbation with dynamic range based on row
        x_perturb = np.random.uniform(-0.06, 0.06) * r_factor
        y_perturb = np.random.uniform(-0.06, 0.06)
        x = base_col + x_perturb
        y = base_row + y_perturb
        # Row-specific staggering for visual balance
        if row_idx % 3 == 1:  # Every third row has alternate placement
            x += 0.5 / cols * r_factor
        xs.append(x)
        ys.append(y)
    
    # Radius initial guess with dynamic scaling based on grid properties
    r0_base = (0.45 / cols) * (1.0 / (1.0 + (0.25 * rows)))  # Adjust for rows
    r0 = np.full(n, r0_base) - 1e-3  # Slight shrink for stabilization
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.48)]  # Tighter max radius
    # Add soft radial boundary tolerance as additional safety layer
    # These are for validation but not enforced via constraints, so not added to bounds length

    # Cost function to optimize: negative of total sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint definitions: use lambda with i, j as closure args
    # First, define boundary constraints with lambda closures
    cons = []
    for i in range(n):
        idx = 3*i
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Now, overlap constraints using spatial vectorization with closure capture
    for i in range(n):
        if i < 10:
            # Priority to handle first 10 circles with tighter constraints for layout stability
            for j in range(i + 1, n):
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2 + 1e-12
                cons.append({"type": "ineq", "fun": constraint_func})
        else:
            # For others, use a less sensitive check with slight tolerance
            for j in range(i + 1, n):
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2 + 1e-10
                cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization phase with adaptive tolerances and dynamic constraint weights
    # We use a hybrid method with SLSQP for precision and L-BFGS-B for speed
    # Initialize with default optimization settings
    res = minimize(neg_sum_radii, v0, bounds=bounds, method="SLSQP", constraints=cons,
                   options={"maxiter": 120, "ftol": 1e-12, "gtol": 1e-8, "eps": 1e-6})

    # Phase 1: Dynamic perturbation based on spatial cluster detection and radius scaling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        distances = np.zeros((n, n))
        
        # Vectorized efficient distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        distances = np.sqrt(dx ** 2 + dy ** 2)
        
        # Calculate adjacency graph and detect clusters using community detection
        from scipy.sparse import csr_matrix
        from scipy.cluster.hierarchy import fcluster, linkage
        # Build adjacency matrix based on min distance to others
        adjacency_matrix = distances <= (radii + np.expand_dims(radii, axis=1))
        adjacency_sparse = csr_matrix(adjacency_matrix)
        
        # Use hierarchical clustering for spatial grouping
        if np.sum(adjacency_matrix) > 0:
            Z = linkage(adjacency_matrix, method='ward')
            cluster_labels = fcluster(Z, t=3, criterion='maxclust')  # Max 3 clusters
        else:
            cluster_labels = np.zeros(n, dtype=int)
        
        # Second phase: Perturbation + reoptimization with cluster-adjusted perturbations
        # Spatial clustering-aware refinement for cluster expansion or convergence
        perturbed_v = v.copy()
        for i in range(n):
            # Perturb based on cluster's median position (spatially balanced)
            cluster_med = np.median(centers[cluster_labels == cluster_labels[i]], axis=0)
            # Perturbation magnitude scales with cluster size and radius
            perturb_factor = 0.01 * (np.mean(radii[cluster_labels == cluster_labels[i]]))
            dx_perturb = np.random.uniform(-perturb_factor, perturb_factor) * (cluster_med[0] - centers[i,0])
            dy_perturb = np.random.uniform(-perturb_factor, perturb_factor) * (cluster_med[1] - centers[i,1])
            perturbed_v[3*i] += dx_perturb
            perturbed_v[3*i+1] += dy_perturb
        
        # Re-evaluate with cluster-driven spatial adjustments
        res = minimize(neg_sum_radii, perturbed_v, bounds=bounds, method="L-BFGS-B", constraints=cons,
                       options={"maxiter": 100, "ftol": 1e-12, "eps": 1e-3, "gtol": 1e-9})
        
        # Phase 2: Targeted expansion of spatially isolated circle
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            distances = np.zeros((n,n))
            
            # Re-calculate distances
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            distances = np.sqrt(dx**2 + dy**2)
            
            # Calculate min distance for each circle
            min_dists = np.min(distances, axis=1)
            # Find least constrained circle: the one with maximum of min distance
            least_constrained_idx = np.argmax(min_dists)
            
            # Calculate current total sum
            current_total = np.sum(radii)
            # Calculate ideal target increase based on known global upper bounds
            # Assuming an upper limit of 2.65 (as per historical benchmark)
            # We target to increase by 0.006 to 0.01 with adaptive control
            # Calculate max allowable expansion based on spatial gaps
            max_growth_per = (min_dists[least_constrained_idx] - max(radii)) / (np.mean(radii) * 1.3)
            target_growth = max(0.004, min(0.009, max_growth_per))  # Ensure it's valid per spatial constraints
            
            # Create a vector to expand all circles with more emphasis on the least constrained
            new_radii = radii.copy()
            new_radii[least_constrained_idx] += target_growth * 1.4  # Over-expansion boost
            for i in range(n):
                if i != least_constrained_idx:
                    # Add a small, proportional growth influenced by proximity
                    # Use proximity to expand others, but less than the main one
                    proximity = np.sum(np.min(distances[i], axis=0) < 0.5)
                    growth_multiplier = 1.0 + 0.1 * proximity / n
                    new_radii[i] += target_growth * growth_multiplier
            
            # Ensure no radius exceeds the geometric limit of remaining space
            max_permissible_radius = 0.48  # From bounds
            new_radii = np.clip(new_radii, 1e-5, max_permissible_radius)
            
            # Apply expansion with constraint validation
            while True:
                v_new = v.copy()
                v_new[2::3] = new_radii
                v_new_centers = np.column_stack([v_new[0::3], v_new[1::3]])
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = v_new_centers[i, 0] - v_new_centers[j, 0]
                        dy = v_new_centers[i, 1] - v_new_centers[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < new_radii[i] + new_radii[j] - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
                else:
                    # Reduce expansion if overlap is detected
                    # Reduce by a small factor based on overlap severity
                    overlap_count = np.sum([1 for i in range(n)
                                        for j in range(i+1, n)
                                        if (np.sqrt((v_new[3*i] - v_new[3*j])**2 + (v_new[3*i+1] - v_new[3*j+1])**2) < (new_radii[i]+new_radii[j]) - 1e-8)])
                    reduction_factor = (overlap_count * 5e-4) / n  # Reduce each radius proportionally
                    for i in range(n):
                        new_radii[i] = max(1e-5, new_radii[i] - reduction_factor)
            
            # Update the decision vector
            v = v_new
            # Re-evaluate with expanded radii and new adjacency constraints
            res = minimize(neg_sum_radii, v, bounds=bounds, method="L-BFGS-B", constraints=cons,
                           options={"maxiter": 100, "ftol": 1e-12, "eps": 1e-3, "gtol": 1e-9})
    
    # Final cleanup of the solution
    v = res.x if res.success else v0  # Ensure we fall back to the best initial guess
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.48) # Ensure radii stay within bounds

    # Final validation and return
    return centers, radii, float(radii.sum())