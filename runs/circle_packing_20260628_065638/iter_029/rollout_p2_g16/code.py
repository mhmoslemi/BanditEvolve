import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with a more refined spatial hashing and geometric constraint-aware initialization
    grid_x = (np.arange(cols) + 0.5) / cols
    grid_y = (np.arange(rows) + 0.5) / rows
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = grid_x[col]
        y_center = grid_y[row]
        
        # Spatial constraint-aware perturbation: shift based on grid position
        x_perturb = np.random.uniform(-0.03, 0.03)
        y_perturb = np.random.uniform(-0.03, 0.03)
        
        # Staggered odd rows: shift alternate rows right to avoid overlap
        if row % 2 == 1:
            x_center += 0.5 / cols * np.random.uniform(0.9, 1.1)  # slight adaptive shift for divergence
        
        x = np.clip(x_center + x_perturb, 0.0, 1.0)
        y = np.clip(y_center + y_perturb, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    
    # Compute base radius considering packing efficiency + spatial awareness
    base_radius = 0.36 / cols
    if rows > cols:
        base_radius *= np.sqrt(cols / rows)  # better alignment in wider aspect ratio
    
    r0 = base_radius - 1e-3  # margin to prevent singularity in solver
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Bounds must be consistent with 3n entries
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-5, 0.5))  # radius
    
    # Objective: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraint definitions
    cons = []
    # Vectorized boundary constraints
    for i in range(n):
        # Left: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right: 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top: 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized circle-circle overlap constraints: distance^2 - (r1 + r2)^2 >= 0
    for i in range(n):
        for j in range(i+1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})
    
    # Initial optimization with enhanced tolerances and aggressive iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP",
                   bounds=bounds,
                   constraints=cons,
                   options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-12})
    
    # First reconfiguration: spatial perturbation with gradient-aware directional expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Spatial constraint-aware gradient-aware perturbation
        # Compute gradient influence vectors (based on constraint satisfaction and distance)
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        
        # Generate directional hash with consideration to distance from boundary and peers
        spatial_hash = np.random.rand(n, 2)
        hash_scales = 0.025 + 0.002 * np.mean(np.clip(radii, 1e-5, 0.5))
        perturbed_v = v.copy()
        for i in range(n):
            # Spatial perturbation with soft gradient influence
            # Use direction perpendicular to nearest peer
            nearest_idx = np.argmin(dists[i, :n]) if n > 1 else 0
            dx_peer = centers[nearest_idx, 0] - centers[i, 0]
            dy_peer = centers[nearest_idx, 1] - centers[i, 1]
            norm = np.sqrt(dx_peer**2 + dy_peer**2)
            if norm < 1e-5:
                dx_peer = np.random.rand() * 0.2
                dy_peer = np.random.rand() * 0.2
            else:
                dx_peer /= norm
                dy_peer /= norm
            # Perpendicular vector to peers for perturbation
            dx_perturb = -dy_peer * (radii[i] / np.mean(radii) * hash_scales * np.random.rand(1))
            dy_perturb = dx_peer * (radii[i] / np.mean(radii) * hash_scales * np.random.rand(1))
            perturbed_v[3*i] += dx_perturb
            perturbed_v[3*i + 1] += dy_perturb
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP",
                       bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-12})
    
    # Final aggressive reconfiguration with spatial reorganization and targeted expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix and determine most constrained (least influenced) circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dists[i, j] = np.sqrt(dx**2 + dy**2)
        
        # Compute spatial influence index for each circle
        influence = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    influence[i] += min(1.0 / dists[i,j], 50) if dists[i,j] > 0 else 0
        
        # Find the most spatially unconstrained circle (least influence)
        unconstrained_idx = np.argmin(influence)
        
        # Spatial reconfiguration: reposition the unconstrained circle to a novel location
        # New position: based on available space between peers
        new_x = np.random.uniform(0.0 + 0.005, 1.0 - 0.005)
        new_y = np.random.uniform(0.0 + 0.005, 1.0 - 0.005)
        
        # Ensure the new position doesn't violate boundaries and maintain spacing
        nearest_distance = np.inf
        for j in range(n):
            dx = new_x - centers[j, 0]
            dy = new_y - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < nearest_distance:
                nearest_distance = dist
        
        if nearest_distance > 0.5:
            # Safe to place
            v[3*unconstrained_idx] = new_x
            v[3*unconstrained_idx + 1] = new_y
            v[3*unconstrained_idx + 2] = radii[unconstrained_idx] * 1.15  # expand based on constraint vacuum
        
        # Now perform second optimization on updated configuration
        res = minimize(neg_sum_radii, v, method="SLSQP",
                       bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-12})
    
    # Final configuration
    # If solver fails, fallback to best so far
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())