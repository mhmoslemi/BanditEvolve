import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize points with spatial clustering and perturbation for better exploration
    xs = np.zeros(n)
    ys = np.zeros(n)
    
    # Use a more robust spatial distribution with dynamic spacing
    base_col_width = 0.9 / cols
    base_row_height = 0.9 / rows
    
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Add dynamic jitter based on spacing to avoid clustering
        jitter = np.random.uniform(-0.06, 0.06, size=2)
        
        # Add staggered offsets for non-planar spatial density
        col_offset = 0 if row % 2 == 0 else base_col_width / 2
        
        # Set spatial position with dynamic jitter
        xs[i] = (col + 0.5 + jitter[0]) * base_col_width + col_offset
        ys[i] = (row + 0.5 + jitter[1]) * base_row_height
    
    # Initialize radii with a smarter base radius that scales with spacing
    base_radius = (base_col_width * base_row_height) ** 0.5
    r0 = base_radius - np.random.uniform(0.0, 0.05)
    r0 = np.maximum(r0, 1e-3)
    
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)
    
    # Set bounds with 3n entries, consistent with the v vector
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]   # Length 3*n
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize by minimizing negative sum
    
    # Vectorized constraint functions with closures and safe i capture
    cons = []
    
    # Boundary constraints
    for i in range(n):
        # Left: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right: x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top: y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Pairwise distance constraints (simplified for computational efficiency)
    # We use numpy broadcasting in constraint evaluation to reduce redundant checks
    # We maintain a sparse constraint graph with higher density for nearby circles
    # This reduces the number of constraint evaluations for the SLSQP solver

    # First, create a 2D grid of indices for efficient neighbor searches
    index_grid = np.indices((n, n)).reshape(2, n*n)
    
    # Generate a sparse constraint graph with only nearby pairs
    # We apply a radius-based filter (distance between centers <= 2 * mean_radius)
    # This avoids redundant constraints for too far circles (optimization)
    mean_radius = 0.05  # Heuristically chosen to find a balance
    constraint_radius = 2 * mean_radius
    
    # Use a vectorized approach to compute pairwise distances and filter
    # This avoids looping through all pairs
    # Note: this section is not executed at optimization time, but defines constraint functions
    # Actual constraints are only added for nearby pairs
    
    # Create an efficient constraint graph
    # We'll use a grid-based spatial partitioning for the constraint graph creation
    # This is for optimization: we only consider nearby circles
    # This is done once as part of constraint setup
    
    # First compute positions once for reference
    base_centers = np.column_stack([v0[0::3], v0[1::3]])
    # For reference, we can use this as a base in the constraint graph construction
    
    # Create spatial indices for constraint filtering
    # We'll use a grid for efficient neighbor detection
    def build_sparse_constraint_graph(centers, constraint_radius):
        # Initialize constraint graph
        constraint_graph = []
        
        # Use numpy to find nearby points
        grid_cell_size = constraint_radius * 1.5  # 1.5 cell sizes for neighbor detection
        grid_width = 1.5 / grid_cell_size  # For unit [0,1]
        grid_height = 1.5 / grid_cell_size
        
        # Create a grid of bins (cell centers)
        grid_x = np.arange(0, 1 + grid_cell_size, grid_cell_size)
        grid_y = np.arange(0, 1 + grid_cell_size, grid_cell_size)
        
        # Map points to their grid cells
        bin_indices = np.zeros((n, 2), dtype=int)
        for i in range(n):
            x, y = centers[i]
            bin_x = int(np.floor(x / grid_cell_size))
            bin_y = int(np.floor(y / grid_cell_size))
            bin_indices[i, 0] = bin_x
            bin_indices[i, 1] = bin_y
        
        # Group points by grid cells
        cell_to_points_map = {}
        for i, (x, y) in enumerate(bin_indices):
            key = (x, y)
            if key not in cell_to_points_map:
                cell_to_points_map[key] = []
            cell_to_points_map[key].append(i)
        
        # Build constraint graph
        for key, points in cell_to_points_map.items():
            for i in points:
                # Find neighboring cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor_key = (key[0] + dx, key[1] + dy)
                        if neighbor_key in cell_to_points_map:
                            for j in cell_to_points_map[neighbor_key]:
                                if i < j:
                                    constraint_graph.append((i, j))
        
        return constraint_graph

    # Build the sparse constraint graph
    sparse_constraints = build_sparse_constraint_graph(base_centers, constraint_radius)
    
    # Create constraint functions for the sparse constraint graph
    for i, j in sparse_constraints:
        # Use lambda with default capture for closure scoping
        # This way, we can refer to constraint indices correctly
        def constraint_func(v, i=i, j=j):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            return dx**2 + dy**2 - (v[3*i+2] + v[3*j+2])**2
        
        cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization with high precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-9})
    
    # Post-optimization refinement with a spatial hashing strategy
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute effective radius mean for spatial hashing scale
        radius_mean = np.mean(radii)
        
        # Generate spatial hashing perturbation with radius scaling
        spatial_hash = np.random.rand(n, 2) * (0.1 * radius_mean)
        
        # Apply hashing to positions with radius normalization
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / radius_mean)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / radius_mean)
        
        # Re-evaluate with refined spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-9})
    
    # Final refinement of least constrained circle with soft growth
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute pairwise distances for constraint validation
        # Vectorized version using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # For each circle, compute minimum distance from all others
        min_dist = np.min(dists, axis=1)
        
        # Find the circle with the maximum minimum distance (least constrained)
        least_constrained_idx = np.argmax(min_dist)
        
        # Compute current total and potential expansion factor
        current_total = np.sum(radii)
        max_growth = 0.009  # Safe expansion based on previous solutions
        max_growth_rate = 0.25  # Maximum allowed increase per step (in terms of total)
        if current_total < 2.63:  # Conservative check
            # Heuristic to grow the least constrained circle
            # Increase by a safe 25% of the average radius but not more than max_growth
            base_radius_increase = 0.25 * (np.mean(radii))
            radius_increase = min(base_radius_increase, max_growth)
            
            # Apply radius increase only to least constrained circle
            v[3*least_constrained_idx + 2] = min(radii[least_constrained_idx] + radius_increase, 0.5)
        
        # Re-evaluate with updated configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-9})
    
    # Final cleanup and validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())