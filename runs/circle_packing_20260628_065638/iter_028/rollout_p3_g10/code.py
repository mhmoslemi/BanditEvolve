import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Spatial initialization with geometric diversity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / (cols + 0.3)  # Slightly asymmetric division
        y_center = (row + 0.5) / (rows + 0.2)  # Add vertical asymmetry
        # Add position bias based on row and column
        x_base = np.random.uniform(-0.1, 0.1) * (1.0 / cols) * (col + 0.5)
        y_base = np.random.uniform(-0.1, 0.1) * (1.0 / rows) * (row + 0.5)
        x = x_center + x_base
        y = y_center + y_base
        
        # Add staggered offset for alternate rows
        if row % 2 == 1:
            x += 0.5 / cols * 0.9  # Reduced stagger for more density
        xs.append(x)
        ys.append(y)
    
    r0 = 0.40 / cols - 1e-3  # Larger initial radius with more room
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with proper closure handling
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: x + r <= 1.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: y + r <= 1.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with adaptive scaling
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with captured i,j with delayed binding
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-10, "eps": 1e-8})
    
    # Asymmetric reconfiguration: spatial hashing with adaptive scaling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create spatial hash map with bias toward low-constraint regions
        # We'll use a simple geometric hashing that weights by distance to nearest walls
        dist_to_wall = np.zeros(n)
        for i in range(n):
            x, y = centers[i]
            dist_to_wall[i] = min(x, 1.0 - x, y, 1.0 - y)
        
        # Spatial hashing with asymmetric perturbation
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbation = spatial_hash * (0.5 + 0.2 * dist_to_wall / np.max(dist_to_wall))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation[i, 0]
            perturbed_v[3*i+1] += perturbation[i, 1]
        
        # Re-evaluate with spatial perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})
    
    # Targeted radius expansion on least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Find least constrained circle by maximizing minimum distance to others
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate potential expansion area
        current_total = np.sum(radii)
        target_growth = 0.01  # Double expansion target
        
        # Create expansion vector with gradient descent approach
        new_radii = radii.copy()
        expansion = 0.0
        for _ in range(10):  # Multi-phase expansion
            # Perturb the least constrained circle slightly
            new_radii[least_constrained_idx] += expansion * 1.02
            # Check validity of this perturbation
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < (new_radii[i] + new_radii[j]) - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                expansion += 0.001
            else:
                break
        
        # Create perturbed version for optimization
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())