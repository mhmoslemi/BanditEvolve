import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Seed based on deterministic hash to ensure reproducibility of spatial layout
    np.random.seed(int(np.prod([cols, rows, n])) % 1000000)
    random_offset = np.random.rand(n, 2) * 0.04

    # Initialize with staggered grid with controlled spatial diversity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce directional randomness to break alignment, especially for even rows
        x = x_center + random_offset[i, 0]
        y = y_center + random_offset[i, 1]
        if row % 2 == 1:  # Stagger alternate rows
            x += 0.5 / cols * np.random.choice([-1, 1]) * 0.6  # More intense staggering
        xs.append(x)
        ys.append(y)
    
    # Dynamic base radius calculation based on packing density and geometric constraints
    # Start with more aggressive initial radius to allow expansion
    base_radius = 0.46 / cols  # Increased base from prior to promote early expansion
    r0 = np.full(n, base_radius - 1e-2)  # Subtract a small value to allow optimizer to grow
    
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0
    
    # Ensure bounds list has exactly 3*n elements
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.48)]  # Slight relaxation to improve packing

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize sum of radii

    # Vectorized constraints using lambda capture with fixed i,j
    cons = []
    for i in range(n):
        # Left constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})
    
    # Initial optimization with aggressive parameter tuning
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-8, "disp": False})
    
    # Primary refinement: asymmetrical spatial perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate enhanced perturbation vector for spatial reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.01  # Tighter spatial perturbation
        perturbed_v = v.copy()
        for i in range(n):
            # Apply directional perturbation based on radius-to-mean ratio
            scale = radii[i] / np.mean(radii)
            perturbed_v[3*i] += spatial_hash[i, 0] * scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale
        
        # Second-stage optimization with tightened constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-8, "disp": False})
    
    # Second-level refinement: advanced spatial analysis and targeted expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorize distance matrix for efficient spatial analysis
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute constrained indices - find least constrained and most constrained
        min_dists = np.min(dists, axis=1)
        max_dists = np.max(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        most_constrained_idx = np.argmin(max_dists)
        
        # Compute growth potential using weighted expansion
        current_total = np.sum(radii)
        growth_factor = 0.008  # Adjusted target growth
        expansion_scale = growth_factor / (n - 1) * (current_total / np.sum(radii))
        
        # Apply asymmetric expansion with dynamic scaling
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_scale * 1.15  # Aggressive growth on least constrained
        for i in range(n):
            if i != least_constrained_idx:
                # Vary expansion based on distance to most constrained
                distance_to_most = np.sqrt((centers[i,0] - centers[most_constrained_idx,0])**2 +
                                         (centers[i,1] - centers[most_constrained_idx,1])**2)
                # Apply stochastic component based on distance
                expansion_i = expansion_scale * (1.0 + 0.05 * np.exp(-distance_to_most * 0.3))
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation using vectorized and optimized steps
        iterations = 0
        while iterations < 4:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Efficient overlap validation using vectorized distance matrix
            dists_exp = np.sqrt((expanded_centers[:,np.newaxis,0] - expanded_centers[np.newaxis,:,0])**2 +
                               (expanded_centers[:,np.newaxis,1] - expanded_centers[np.newaxis,:,1])**2)
            overlaps = np.where(dists_exp < (new_radii[:,np.newaxis] + new_radii[np.newaxis,:]) - 1e-8, True, False)
            overlap_sum = overlaps.sum()
            
            if overlap_sum == 0:  # No valid overlaps
                break
            else:
                # Reduce expansion proportionally to overlap severity
                new_radii = radii + (new_radii - radii) * (1.0 - overlap_sum / (n*(n-1)/2))
                iterations += 1
        
        # Apply valid expanded radii and proceed
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with tighter tolerance and increased iterations
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-11, "eps": 1e-8, "disp": False})
    
    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())