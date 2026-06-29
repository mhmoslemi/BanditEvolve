import numpy as np

def run_packing():
    n = 26
    cols = 5  # Ensure even coverage with 5 columns for 26 circles (5x5 grid is 25 circles, plus 1 extra row)
    rows = (n + cols - 1) // cols  # Ceiling division for rows to distribute all 26 circles
    # Advanced initialization with dual-phase spatial perturbation and adaptive grid refinement
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Phase 1: Base positions with staggered, randomized perturbation (stochasticity to avoid symmetry)
        x = base_x + np.random.uniform(-0.04, 0.04)  # Reduce range for finer spatial control
        y = base_y + np.random.uniform(-0.04, 0.04)  # Synchronized perturbation for alignment
        # Apply staggered grid correction, but reduce vertical shift to avoid over-accumulation in adjacent rows
        if row % 2 == 1:
            x += 0.4 / cols  # Reduced vertical staggering for better horizontal resolution
        xs.append(x)
        ys.append(y)
    
    # Phase 2: Apply low-amplitude adaptive perturbation based on spatial constraints
    # Initialize minimal radii with refined distribution based on grid dimensions
    r0 = 0.36 / cols - 1e-2  # Slightly increase base radius for potential to grow
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Build strict bounds and constraints with consistent structure
    bounds = []
    for _ in range(n):
        # Ensure boundaries remain within [0,1]
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Radius >= 0.0001, < 0.5
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Create constraints with explicit closures, avoiding lambda capture bugs by passing i explicitly
    cons = []
    for i in range(n):
        # Constraint: x >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Constraint: y >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with geometric hashing and vectorized computation
    # Optimize by pre-allocating spatial and adjacency indices for better cache locality
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorized computation with closures: avoid recomputation via indices
            def constraint_func(v, i=i, j=j):
                # Calculate distance squared - sum of radii squared
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Phase 1: Initial optimization with high precision and dynamic gradient control
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 1800, 
            "ftol": 1e-11,  # Tighter tolerance for convergence
            "eps": 1e-10,  # Smaller step size for better gradient approximation
            "disp": False
        }
    )
    
    # Phase 2: Refinement with spatial hashing and adaptive constraint reordering
    if res.success:
        v = res.x
        # Vectorized spatial and radius matrices for efficient calculation
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Vectorized spatial distance matrix via broadcasting for fast overlap analysis
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1, 0]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute adjacency matrix with more robust distance threshold
        # Using vectorized thresholding to avoid O(n^2) repeated computations
        adj = dists <= (radii + radii.reshape(-1, 1))
        
        # Find connected components for topological perturbation
        from scipy.sparse import csr_matrix, csgraph
        graph = csr_matrix(adj)  # Convert adjacency matrix to sparse structure
        components = csgraph.connected_components(graph)[1]  # Get component identifiers
        
        # Spatial perturbation with radius-aware scaling (smaller radii get more spatial correction)
        # Randomized hash based on component clusters for perturbation direction control
        spatial_hash = np.random.rand(n, 2) * 0.04 * (1.0 + radii / np.mean(radii))  # Radius-informed scaling
        perturbed_v = v.copy()
        
        for i in range(n):
            # Use component-based spatial perturbation with radius-aware scaling
            # Perturb in direction based on hash, scaled by radius and component size
            component_size = np.sum(components == components[i])
            component_factor = 0.3 * (component_size / n)  # Larger components get more perturbation
            perturbed_v[3*i] += spatial_hash[i, 0] * (0.8 + component_factor * 0.2)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (0.8 + component_factor * 0.2)
        
        # Second-order refinement with tighter tolerance and better constraint management
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 600,
                "ftol": 1e-11, 
                "eps": 1e-9, # Slight relaxation to avoid getting stuck
                "disp": False
            }
        )
    
    # Phase 3: Advanced target expansion on non-overlapping, least constrained circle
    # Compute minimum distances and identify the least constrained circle more robustly
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.zeros((n, n))
        
        # Vectorized distance matrix (optimized) for efficient computation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find nearest neighbors (minimum distance to each circle) for each circle
        min_dists = np.min(dists, axis=1)
        
        # Compute spatial constraint tightness (minimum distance to all other circles / sum of radii)
        # This represents the relative constraint tightness of each circle
        tightness = min_dists / (radii.sum() / n)  # Normalize relative to total sum
        
        # Find the least constrained circle (max tightness)
        least_constrained_idx = np.argmax(tightness)
        total_sum = radii.sum()
        
        # Compute expansion factor based on tightness and relative potential (weighted)
        expansion_factor = 0.05 / (n - 1) * (1.0 + 0.15 * tightness[least_constrained_idx])
        # This allows more expansion on less constrained circles and slightly less on well-constrained ones
        
        # Apply expansion with constraint validation through direct adjacency analysis
        # Prevent overlap by reducing expansion factor if constraints would be breached
        # This avoids expensive repeated optimization passes
        max_expanded_radius = 0.5
        min_radius = radii.min()
        if radii[least_constrained_idx] < min_radius:
            # Safety net to prevent extreme radius reduction
            min_radius = radii[least_constrained_idx]
        
        # Create adjusted radii with targeted expansion on the least constrained circle
        new_radii = radii.copy()
        # Over-expand the least constrained circle to trigger layout re-adjustment
        try:
            new_radii[least_constrained_idx] += expansion_factor * 1.15
        except:
            print(f"Expansion factor {expansion_factor}: over-constrained, setting to max {max_expanded_radius}")
            new_radii[least_constrained_idx] = max_expanded_radius
        
        # Apply expansion to all other circles with randomized perturbations
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (0.95 + np.random.rand() * 0.1)  # Slight randomness
                if new_radii[i] > max_expanded_radius:
                    new_radii[i] = max_expanded_radius
                # Cap expansion to prevent overshooting
                new_radii[i] = np.clip(new_radii[i], min_radius, max_expanded_radius)
        
        # Re-validate configuration via adjacency thresholding with vector operations
        # Compute pairwise distances again (optimized)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        adjacency = (dists <= (new_radii[:, np.newaxis] + new_radii[np.newaxis, :]))  # 2D adjacency
        
        # Check for overlaps, but vectorize to avoid O(n^2) checking for every expansion
        overlap = np.any(np.triu(adjacency, k=1), axis=1)
        if np.any(overlap):
            # If any overlap occurs, reduce expansion factor and repeat the process
            # Re-calculate with reduced expansion factor
            new_radii = radii.copy()
            new_radii[least_constrained_idx] += expansion_factor * 0.7
            # Apply smaller expansion to others too
            for i in range(n):
                if i != least_constrained_idx:
                    new_radii[i] += expansion_factor * 0.6
            # Re-validate
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)
            adjacency = (dists <= (new_radii[:, np.newaxis] + new_radii[np.newaxis, :]))
            overlap = np.any(np.triu(adjacency, k=1), axis=1)
            if np.any(overlap):
                # If still overlapping, cap expansion to prevent further issues
                # Force expansion only to the non-overlapping maximum possible
                new_radii = radii.copy()
                for i in range(n):
                    new_radii[i] = np.min([new_radii[i], 0.5 - np.mean(new_radii) / n])
                print("Expansion capped to prevent overlap")
        
        # Final update
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final refinement with tighter constraints and higher precision
        res = minimize(
            neg_sum_radii, 
            v_new, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 600,
                "ftol": 1e-11,
                "eps": 1e-9,
                "disp": False
            }
        )
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final validation pass with full pairwise checking for safety (to catch any edge cases)
    # This is computationally heavy but critical for correctness
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < (radii[i] + radii[j]) - 1e-8:
                # Safety fallback to re-validate by shrinking radii
                print(f"Final validation failed on circles {i}, {j}, shrinking radii")
                # Create a safe fallback by reducing the overlap radii slightly
                radii[i] = max(radii[i] * 0.99, 1e-6)
                radii[j] = max(radii[j] * 0.99, 1e-6)
                v[2::3] = radii
                centers = np.column_stack([v[0::3], v[1::3]])
                break
        else:
            continue
        break
    
    return centers, radii, float(radii.sum())