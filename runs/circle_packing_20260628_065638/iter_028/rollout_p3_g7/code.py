import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with staggered and scaled grid with randomized distortion
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add directional perturbation proportional to column and row distances
        x = x_center + np.random.uniform(-0.02, 0.02) * (0.25 * (col + 1))
        y = y_center + np.random.uniform(-0.02, 0.02) * (0.25 * (row + 1))
        if row % 2 == 1:
            x += 0.5 / cols * (0.5 + np.random.uniform(-0.25, 0.25))
        xs.append(x)
        ys.append(y)
    
    # Start with smaller radius, adjusted for grid layout and spacing
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Construct proper bounds (3 entries per circle, 3*26)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # x, y, radius bounds
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize sum of radii
    
    # Construct all the constraints explicitly and cleanly
    cons = []
    for i in range(n):
        x, y, r = (3*i, 3*i+1, 3*i+2)
        # Left boundary constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, x=x, r=r: v[x] - v[r]})
        # Right boundary constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, x=x, r=r: 1.0 - v[x] - v[r]})
        # Bottom boundary constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, y=y, r=r: v[y] - v[r]})
        # Top boundary constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, y=y, r=r: 1.0 - v[y] - v[r]})
    
    # Construct overlap constraints with broadcasted geometry
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx * dx + dy * dy - (v[3*i+2] + v[3*j+2]) ** 2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Stage 1: Initial optimization with increased precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-10})
    
    if res.success:
        v = res.x
        # Stage 2: Asymmetric reconfiguration with spatial-aware perturbation
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create a spatial-aware perturbation matrix based on distance to neighbors
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_distances = np.min(dists, axis=1)
        normalization = np.mean(min_distances) + 1e-12  # to avoid division by zero
        
        # Create perturbation that scales based on distance and radius
        perturbation_scale = np.sqrt(radii) * (1 + 0.5 * (min_distances / normalization))
        spatial_perturbation = np.random.rand(n, 2) * perturbation_scale[:, np.newaxis]
        
        # Apply perturbation
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_perturbation[i, 0]
            perturbed_v[3*i+1] += spatial_perturbation[i, 1]
        
        # Stage 2.1: Reoptimize with perturbed spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-11})
        
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Stage 3: Targeted expansion of the least constrained circle
            # Re-compute all pairwise distances for robustness
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx ** 2 + dy ** 2)
            min_dists = np.min(dists, axis=1)
            
            # Find circle with the largest minimal distance (least constrained)
            least_constrained_idx = np.argmax(min_dists)
            max_min_distance = min_dists[least_constrained_idx]
            
            # Determine potential for expansion
            current_total = np.sum(radii)
            # Try to increase the total sum by 0.008 (increase from 2.634292 to 2.642)
            target_total = current_total + 0.008
            expansion_factor = (target_total - current_total) / (n - 1)
            
            # Apply exponential expansion to least constrained circle and neighboring circles
            for i in range(n):
                # Use a non-uniform expansion based on proximity and distance
                if i == least_constrained_idx:
                    # Exponentially grow the least constrained circle
                    v[3*i+2] += expansion_factor * 1.2  # Slight overexpansion
                else:
                    # Adjust expansion based on relative proximity
                    dist_to_least = dists[i, least_constrained_idx]
                    scaled_factor = (dist_to_least / max_min_distance) ** 0.7
                    v[3*i+2] += expansion_factor * scaled_factor * 0.8
                    
            # Stage 3.1: Final optimization with controlled expansion
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())