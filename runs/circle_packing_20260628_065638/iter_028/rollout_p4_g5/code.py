import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    rows = np.ceil(n / cols).astype(int)
    
    # Use adaptive placement with more randomness to break symmetry and explore deeper configurations
    xs, ys = [], []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.4) / cols * 0.95  # Slightly shift base to reduce edge clustering
        base_y = (row + 0.4) / rows * 0.95
        x = base_x + np.random.uniform(-0.08, 0.08) * (rows / (n ** 0.4))
        y = base_y + np.random.uniform(-0.08, 0.08) * (cols / (n ** 0.4))
        # Alternating rows with offset for staggered grid
        if row % 2 == 1:
            x += 0.5 / cols * 0.8
        xs.append(x)
        ys.append(y)
    
    # Use more refined initial radii based on grid density and spacing
    base_radius = 0.35 / (cols + (rows - cols) * 0.6)
    r0 = base_radius - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Define objective function with gradient to improve optimization
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Define boundary constraints with proper lambda closures to ensure correct indices
    cons = []
    for i in range(n):
        # Left boundary constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: 1 - (x + r) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: 1 - (y + r) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Define all pairwise circle separation constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Create anonymous function that captures i and j to avoid lambda capture issues
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})
    
    # Initial optimization with tight constraints and convergence tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-9})
    
    # Apply shake heuristic: 
    # 1. Find the most constrained circle (least free space)
    # 2. Perturb it slightly and reoptimize
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute free space metrics
        free_space = []
        for i in range(n):
            dx = np.abs(centers[:, 0] - centers[i, 0])
            dy = np.abs(centers[:, 1] - centers[i, 1])
            dists = np.sqrt(dx**2 + dy**2)
            free_space.append(np.min(dists[dists > 1e-12]) - radii[i])
        free_space_arr = np.array(free_space)
        
        # Find circle with the lowest free space (most constrained)
        least_free_idx = np.argmin(free_space_arr)
        
        # Perturb the circle and reoptimize to escape local minima
        perturbed_v = v.copy()
        # Perturb with magnitude proportional to circle size and free space
        perturb_magnitude = max(0.0005, np.sqrt(radii[least_free_idx] * free_space_arr[least_free_idx]))
        shift_x = np.random.uniform(-perturb_magnitude, perturb_magnitude) * (1.0 + 0.1 * np.random.rand())
        shift_y = np.random.uniform(-perturb_magnitude, perturb_magnitude) * (1.0 + 0.1 * np.random.rand())
        
        perturbed_v[3*least_free_idx] += shift_x
        perturbed_v[3*least_free_idx+1] += shift_y
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-9})
    
    # Second-phase refinement with adaptive constraints
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Use more efficient constraint evaluation for optimization
        # Only evaluate constraints when radius changes significantly
        # Compute pairwise distances with vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Create new constraints for circles with small radii that may need expansion
        new_cons = []
        for i in range(n):
            # Left boundary
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            # Right boundary
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            # Bottom boundary
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            # Top boundary
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < radii[i] + radii[j] - 1e-4:
                    # Add more restrictive constraint for overlapping circles
                    new_cons.append({"type": "ineq", 
                                     "fun": lambda v, i=i, j=j: 
                                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                         - (v[3*i+2] + v[3*j+2])**2})
        
        # Refine optimization with refined constraints
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=new_cons, options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-9})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())