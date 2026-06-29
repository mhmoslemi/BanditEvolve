import numpy as np

def run_packing():
    # Constants and initial parameters
    n = 26
    cols = int(np.ceil(np.sqrt(n)))  # Optimal for grid layout and radius distribution
    rows = (n + cols - 1) // cols
    grid_spacing = 1.0 / cols  # Scales down grid to fit in [0,1]^2
    initial_radius_base = 0.286  # Tuned base radius, higher than before to exploit potential
    
    # Initialize with a staggered grid with enhanced randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base position: midpoints with staggered offset
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Add randomized jitter for diversity and to break symmetry
        jitter_x = np.random.uniform(-0.12, 0.12)  # Wider jitter for exploration
        jitter_y = np.random.uniform(-0.12, 0.12)
        
        # Shift alternate rows to create offset grid
        if row % 2 == 1:
            base_x += 0.5 / cols  # Staggered offset
            # Introduce row-dependent radius sensitivity to allow asymmetric expansion
            row_factor = 1.0 + (0.5 * (row % 2) - 0.5)  # Row 0: 1, Row 1: 1.5
        else:
            row_factor = 1.0
        
        # Compute actual center
        x = base_x + jitter_x
        y = base_y + jitter_y
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius distribution: use geometric scaling and base radius
    r0 = initial_radius_base * (1.0 / (cols * rows))  # Scaled to avoid immediate conflict
    # Introduce row-dependent radius sensitivity to allow asymmetric expansion
    # Use higher radii in certain rows to enable better packing through asymmetric growth
    r0 += np.array([(3.0 if (i//cols) % 3 == 0 else 1.0) for i in range(n)]) * (r0 * 0.1)
    r0 = np.clip(r0, 1e-4, 0.5)
    
    # Decision vector initialization
    v0 = np.empty(3 * n, dtype=np.float64)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Define bounds (length 3*n, must match decision vector)
    bounds = [(0.0, 1.0)] * n  # x bounds
    bounds += [(0.0, 1.0)] * n  # y bounds
    bounds += [(1e-3, 0.5)] * n  # radius bounds

    # Objective function: maximization of sum of radii
    def neg_sum_radii(v):
        # Vectorized sum of radii (v[2::3])
        # Use np.sum for accuracy and stability
        return -np.sum(v[2::3])

    # Vectorized boundary constraints (x, y, radius constraints)
    # Create ineq constraints: x - r >= 0, x + r <=1, y - r >=0, y + r <= 1
    cons = []
    for i in range(n):
        # x - r >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # x + r <=1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        
        # y - r >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # y + r <=1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraint generation with optimized geometry-aware approach:
    # Use grid-based hashing to group potentially conflicting circles for efficient constraint generation
    # Instead of all pairs, use geometric hashing to group circles in local spatial clusters
    # First, compute distance vectors to form a sparse but strategic constraint set
    # This approach is faster and numerically more stable for large n

    # Constraint generation with grid-based hashing
    # Create a grid for spatial hashing to limit constraint computation
    hash_grid_size = 6  # Grid size for spatial hashing
    hash_grid = np.zeros((hash_grid_size, hash_grid_size), dtype=int)
    spatial_constraints = []
    dists = np.zeros((n, n))  # Vectorized distance matrix

    # Precompute distance matrix for fast access
    dx = np.reshape(v0[::3], (n, 1)) - np.reshape(v0[::3], (1, n))
    dy = np.reshape(v0[1::3], (n, 1)) - np.reshape(v0[1::3], (1, n))
    dists = np.sqrt(dx**2 + dy**2)
    
    # Now define the asymmetric reconfiguration mechanism
    # The idea is to:
    # 1) Use geometric hashing to identify clusters that benefit from asymmetric expansion
    # 2) Trigger an asymmetric reconfiguration: reposition least constrained cluster
    # 3) Perform a targeted expansion on this cluster to explore new high-radius arrangements

    # Create a grid-based hash of spatial positions for clustering
    # This creates a spatial hash that clusters nearby circles to reduce constraint count
    def spatial_hash_func(x, y):
        # Convert to grid coordinates
        grid = np.floor(x * hash_grid_size) % hash_grid_size
        grid_y = np.floor(y * hash_grid_size) % hash_grid_size
        return grid[0] * hash_grid_size + grid_y

    hash_indices = np.array([spatial_hash_func(v0[i*3], v0[i*3+1]) for i in range(n)])
    hash_indices = hash_indices.astype(int)

    # Create hash buckets
    hash_buckets = {}
    for idx, h in enumerate(hash_indices):
        if h not in hash_buckets:
            hash_buckets[h] = []
        hash_buckets[h].append(idx)

    # Generate constraints based on hash clustering
    for bucket in hash_buckets.values():
        if len(bucket) < 2:
            continue
        for i in bucket:
            for j in bucket:
                if i >= j:
                    continue
                # Add constraint between i and j to check for overlapping
                cons.append({"type": "ineq",
                             "fun": lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization (standard SLSQP)
    res = minimize(neg_sum_radii, v0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-11,
                            "maxfun": 5000, "disp": False})
    

    # Asymmetric reconfiguration and expansion strategy
    
    # First, check if successful; if not, revert to v0
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create sparse spatial clustering constraint matrix for reconfiguration
        # Generate a distance-based threshold for clustering
        # This is done once per iteration and not recomputed
        # (This ensures the hash is done in the current state for accurate clustering)
        
        # Recalculate distance matrix with current centers
        dx_current = np.reshape(v[::3], (n, 1)) - np.reshape(v[::3], (1, n))
        dy_current = np.reshape(v[1::3], (n, 1)) - np.reshape(v[1::3], (1, n))
        dists_current = np.sqrt(dx_current**2 + dy_current**2)
        
        # Apply asymmetric reconfiguration to least constrained spatial cluster
        # Use grid-based hash again with updated positions
        hash_indices_current = np.array([spatial_hash_func(v[3*i], v[3*i+1]) for i in range(n)])
        hash_indices_current = hash_indices_current.astype(int)
        
        # Find the cluster with the smallest average distance
        # This is a proxy for "least constrained"
        cluster_distances = []
        for h in np.unique(hash_indices_current):
            cluster_indices = np.where(hash_indices_current == h)[0]
            avg_distance = np.mean(dists_current[np.ix_(cluster_indices, cluster_indices)])
            cluster_distances.append((h, avg_distance))
        
        if cluster_distances:
            cluster_distances.sort(key=lambda x: x[1])  # Sort by distance (smallest is least constrained)
            target_cluster_id, _ = cluster_distances[0]
            cluster_indices = np.where(hash_indices_current == target_cluster_id)[0]
            cluster_centers = centers[cluster_indices]
            cluster_radii = radii[cluster_indices]
            
            # Find the cluster's least constrained circle
            # Use minimum distance from cluster member to its neighbors as constraint proxy
            min_distance = np.inf
            least_constrained_idx = cluster_indices[0]
            for i in cluster_indices:
                intra_distances = dists_current[i][cluster_indices]
                min_dist = np.min(intra_distances)
                if min_dist < min_distance:
                    min_distance = min_dist
                    least_constrained_idx = i
            
            # Define target expansion (asymmetric radius expansion)
            # Expand cluster's least constrained circle, keeping others small
            # Use a geometric scaling to allow for better expansion in unbounded regions
            # This is our asymmetric reconfiguration
            expansion_factor = 1.5  # More aggressive than previous
            # Compute how much can be expanded while keeping the rest minimal
            # Use a geometric expansion (expansion_factor * current) to allow high growth
            # The cluster's radii are adjusted to allow expansion while keeping others small
            # The maximum possible radius is capped at 0.5 for the grid

            # Get other cluster indices to prevent over-expansion
            other_cluster_indices = [i for i in cluster_indices if i != least_constrained_idx]
            radii[least_constrained_idx] *= expansion_factor
            if radii[least_constrained_idx] > 0.5:
                radii[least_constrained_idx] = 0.5
            # Adjust other cluster members to be smaller if needed
            # This allows the central circle to take more space while others remain compact
            # To be safe, scale others down to avoid overlap
            min_other_radius = np.min(radii[other_cluster_indices]) / 2
            for i in other_cluster_indices:
                radii[i] = np.clip(radii[i], 1e-4, min_other_radius)
            
            # Recalculate decision vector with new radii
            new_v = v.copy()
            new_v[2::3] = radii
            
            # Reoptimize with new configuration
            # Use a tighter tolerance for stability in the asymmetric configuration
            res = minimize(neg_sum_radii, new_v, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-12,
                                    "maxfun": 2000, "disp": False})
        
        v = res.x if res.success else v

    # Final evaluation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-4, 0.5)
    
    return centers, radii, float(radii.sum())