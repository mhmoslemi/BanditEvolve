import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with enhanced spatial sampling and adaptive perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Dynamic perturbation based on row distance and radius potential
        x = x_center + np.random.uniform(-0.07, 0.07) * (1 + 0.5 * np.random.rand())
        y = y_center + np.random.uniform(-0.07, 0.07) * (1 + 0.5 * np.random.rand())
        
        # Adaptive staggered grid for even spacing
        if row % 2 == 1:
            x += 0.5 / cols * (1 + np.random.rand() * 0.2)
        xs.append(x)
        ys.append(y)
    
    # Adaptive initial radius based on spatial distribution and packing density estimation
    avg_dist = np.sqrt((1/cols)**2 + (1/rows)**2) * 0.85
    r0 = (avg_dist / np.sqrt(2)) * 0.8 - 1e-3  # Adjust packing factor for optimized density
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with closed-form expressions
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with dynamic spacing normalization and regularization
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                          "fun": lambda v, i=i, j=j: 
                              (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                              - (v[3*i+2] + v[3*j+2])**2 * (1 + 1e-3 * (v[3*i+2] + v[3*j+2]))})
    
    # Initial optimization with adaptive solver configuration and memory
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP", 
        bounds=bounds,
        constraints=cons,
        options={ 
            "maxiter": 1200, 
            "ftol": 1e-11, 
            "gtol": 1e-11,
            "eps": 1e-10,
            "iprint": 0,
            "disp": False,
            "maxcor": 100,
            "rho": 0.6,
            "sigma": 0.1,
            "finite_diff_rel_step": 1e-6
        }
    )
    
    # Asymmetric reconfiguration with adaptive perturbation based on spatial gradient
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate enhanced adaptive perturbation map based on local density
        spatial_gradients = []
        for i in range(n):
            dx = centers[i, 0] - centers.mean(0)[0]
            dy = centers[i, 1] - centers.mean(0)[1]
            grad = np.sqrt(dx**2 + dy**2) / np.sqrt(n)
            spatial_gradients.append(grad)
        
        # Constructing spatial perturbation based on inverse gradient for high density regions
        perturbation = []
        for i in range(n):
            scale = 1.0 + 0.5 * (1 - spatial_gradients[i] / np.max(spatial_gradients))
            x_perturb = np.random.uniform(-0.09 * scale, 0.09 * scale)
            y_perturb = np.random.uniform(-0.09 * scale, 0.09 * scale)
            perturbation.append((x_perturb, y_perturb))
        
        # Apply perturbation to spatial parameters
        v_perturbed = v.copy()
        for i in range(n):
            v_perturbed[3*i] += perturbation[i][0]
            v_perturbed[3*i+1] += perturbation[i][1]
        
        # Re-evaluate with enhanced perturbation and tighter constraints
        res = minimize(
            neg_sum_radii,
            v_perturbed,
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={ 
                "maxiter": 500, 
                "ftol": 1e-11, 
                "gtol": 1e-11,
                "eps": 1e-10,
                "iprint": 0,
                "disp": False,
                "maxcor": 100,
                "rho": 0.6,
                "sigma": 0.1,
                "finite_diff_rel_step": 1e-6
            }
        )
    
    # Targeted asymmetric expansion leveraging density gradient
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate minimum distance to others for each circle
        pairwise_distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                pairwise_distances[i, j] = np.sqrt(dx*dx + dy*dy)
        
        min_dists = np.min(pairwise_distances, axis=1)
        
        # Identify least constrained circle with gradient-weighted scoring
        weighted_min_ds = min_dists * (1 + 0.5 * (np.mean(radii) / radii))
        least_constrained_idx = np.argmax(weighted_min_ds)
        
        # Estimate potential expansion based on spatial distribution and neighbor constraints
        min_growth_potential = np.inf
        for j in range(n):
            if j != least_constrained_idx:
                current_dist = pairwise_distances[least_constrained_idx][j]
                if current_dist > radii[least_constrained_idx] + radii[j] - 1e-4:
                    # Calculate potential growth with safety margin
                    potential_growth = (current_dist - radii[least_constrained_idx] - radii[j]) * 0.8
                    if potential_growth < min_growth_potential:
                        min_growth_potential = potential_growth
        
        # Apply expansion with controlled growth and gradient-based allocation
        expansion_factor = min_growth_potential * (0.9 + np.random.rand() * 0.2)
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor
        
        # Apply distributed growth to all circles based on inverse distance to center
        center_point = np.mean(centers, axis=0)
        for i in range(n):
            if i != least_constrained_idx:
                dx = centers[i, 0] - center_point[0]
                dy = centers[i, 1] - center_point[1]
                dist_to_center = np.sqrt(dx**2 + dy**2)
                if dist_to_center > 1e-7:
                    new_radii[i] += expansion_factor * 0.1 * (3 / dist_to_center)
        
        # Constraint-aware expansion with safe gradient descent
        total_sum = np.sum(new_radii)
        target_sum = total_sum + (np.mean(new_radii) * 0.03)
        new_radii = new_radii * (target_sum / np.sum(new_radii))
        
        # Apply new radii with gradient constraint validation
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        res = minimize(
            neg_sum_radii,
            expanded_v,
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={ 
                "maxiter": 400, 
                "ftol": 1e-11, 
                "gtol": 1e-11,
                "eps": 1e-10,
                "iprint": 0,
                "disp": False,
                "maxcor": 100,
                "rho": 0.6,
                "sigma": 0.1,
                "finite_diff_rel_step": 1e-6
            }
        )
    
    # Final refinement with density-aware spatial adjustment
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate spatial density gradient for refined adjustment
        grid_density = np.zeros(n)
        for i in range(n):
            cx, cy = centers[i]
            dx = np.abs(cx - np.mean(centers[:, 0]))
            dy = np.abs(cy - np.mean(centers[:, 1]))
            grid_density[i] = 1.0 / (dx + dy + 1e-7)  # Avoid division by zero
        
        # Apply density-aware spatial adjustment
        v_refined = v.copy()
        for i in range(n):
            if grid_density[i] > np.median(grid_density):
                # Higher density regions get slight compression
                v_refined[3*i] *= 1.0 - 0.01 * grid_density[i]
                v_refined[3*i+1] *= 1.0 - 0.01 * grid_density[i]
            else:
                # Lower density regions get slight expansion
                v_refined[3*i] *= 1.0 + 0.005 * grid_density[i]
                v_refined[3*i+1] *= 1.0 + 0.005 * grid_density[i]
        
        res = minimize(
            neg_sum_radii,
            v_refined,
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={ 
                "maxiter": 300, 
                "ftol": 1e-11, 
                "gtol": 1e-11,
                "eps": 1e-10,
                "iprint": 0,
                "disp": False,
                "maxcor": 100,
                "rho": 0.6,
                "sigma": 0.1,
                "finite_diff_rel_step": 1e-6
            }
        )
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())