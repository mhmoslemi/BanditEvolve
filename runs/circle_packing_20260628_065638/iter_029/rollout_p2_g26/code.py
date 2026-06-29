import numpy as np

def run_packing():
    n = 26
    cols = int(np.sqrt(n)) + 1  # 6 for 26, to distribute better
    rows = (n + cols - 1) // cols
    
    # Base grid with asymmetric spatial hashing to break symmetry
    grid_x_centers = np.arange(cols) / (cols - 0.01)  # Slight skew for better spread
    grid_y_centers = np.arange(rows) / (rows - 0.01)
    
    # Generate initial centers with dynamic perturbation and asymmetry
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        # Base grid position
        x = grid_x_centers[col] + np.random.uniform(-0.07, 0.07) * np.sin(np.pi * col / cols)
        y = grid_y_centers[row] + np.random.uniform(-0.07, 0.07) * np.sin(np.pi * row / rows)
        
        # Stagger rows asymmetrically to avoid alignment
        # Even rows: shift to the right
        # Odd rows: shift to the left, but with varying magnitude
        shift = 0.0
        if row % 2 == 0:
            shift = 0.5 / cols * (1 + np.random.uniform(-0.1, 0.1))
        elif row % 2 == 1:
            shift = -0.5 / cols * (1 + np.random.uniform(-0.1, 0.1))
        x += shift
        
        # Ensure boundaries
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation with improved scaling and spatial sensitivity
    radius_base = 0.32 / cols * (1 + 0.5 * np.random.rand()) - 1e-3
    r0 = radius_base * np.ones(n)
    
    # Decision vector and bounds initialization
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # Create bounds array of size 3*n
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-4, 0.5))  # Minimum radius to prevent singularities
    
    # Objective function: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraint definitions
    # These are vectorized and will be optimized with closure capturing
    
    # Define boundary constraints per circle
    # We'll create a more precise formulation using a list for cleaner handling
    cons = []
    
    # Define all boundary constraints for all circles
    for i in range(n):
        x_i = v0[3*i]
        y_i = v0[3*i + 1]
        r_i = v0[3*i + 2]
        
        # Left boundary: x - r >= 0
        cons.append({
            "type": "ineq", 
            "fun": lambda v, idx=i: v[3*idx] - v[3*idx + 2]
        })
        # Right boundary: x + r <= 1
        cons.append({
            "type": "ineq", 
            "fun": lambda v, idx=i: 1.0 - v[3*idx] - v[3*idx + 2]
        })
        # Bottom boundary: y - r >= 0
        cons.append({
            "type": "ineq", 
            "fun": lambda v, idx=i: v[3*idx + 1] - v[3*idx + 2]
        })
        # Top boundary: y + r <= 1
        cons.append({
            "type": "ineq", 
            "fun": lambda v, idx=i: 1.0 - v[3*idx + 1] - v[3*idx + 2]
        })
    
    # Define overlap constraints (distance^2 - (r_i + r_j)^2 >= 0)
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                "type": "ineq", 
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j])**2 + (v[3*i + 1] - v[3*j + 1])**2 
                    - (v[3*i + 2] + v[3*j + 2])**2
                )
            })
    
    # Initial optimization: enhanced configuration with more robust parameters
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds,
        constraints=cons,
        options={ 
            "maxiter": 1000,
            "ftol": 1e-11,
            "gtol": 1e-11,
            "eps": 1e-10,
            "disp": False,
            "iprint": 0
        }
    )
    
    # If initial optimization fails, use a more adaptive spatial perturbation strategy
    if not res.success:
        print("Initial optimization failed - initiating secondary configuration")
        
        # Redefine v0 with more adaptive initial distribution
        xs = []
        ys = []
        for i in range(n):
            col = i % cols
            row = i // cols
            # Generate base grid with adaptive spread
            x_base = grid_x_centers[col] + np.random.uniform(-0.07, 0.07)
            y_base = grid_y_centers[row] + np.random.uniform(-0.07, 0.07)
            
            # Add row-based perturbation for asymmetry
            row_perturbation = 0.25 / cols * np.sin(2 * np.pi * row / rows) * np.random.uniform(0.5, 1.0)
            if row % 2 == 0:
                x_base += row_perturbation
            else:
                x_base -= row_perturbation
            
            # Ensure boundaries
            x_base = np.clip(x_base, 0.0, 1.0)
            y_base = np.clip(y_base, 0.0, 1.0)
            xs.append(x_base)
            ys.append(y_base)
        
        v0 = np.empty(3 * n)
        v0[0::3] = np.array(xs)
        v0[1::3] = np.array(ys)
        v0[2::3] = radius_base * np.ones(n)
        
        # Re-run with same constraints with enhanced parameters
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
                "disp": False,
                "iprint": 0
            }
        )
    
    # If still not successful, perform spatial perturbation with dynamic gradient-awareness
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial perturbation vector using geometric properties: radii-aware and gradient-aware
        # Spatial hash vector with soft exponential scaling
        spatial_hash = np.random.rand(n, 2) * 0.04
        gradient_weights = np.sqrt(radii) * np.mean(1.0 / (radii + 1e-8))
        
        # Compute perturbation vector based on spatial hash and radii sensitivity
        perturb_v = v.copy()
        for i in range(n):
            dx = spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * gradient_weights[i] * 0.9
            dy = spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * gradient_weights[i] * 0.9
            perturb_v[3*i] += dx
            perturb_v[3*i + 1] += dy
        
        # Secondary optimization using the new vector
        res = minimize(
            neg_sum_radii, 
            perturb_v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={ 
                "maxiter": 800,
                "ftol": 1e-11,
                "gtol": 1e-11,
                "eps": 1e-10,
                "disp": False,
                "iprint": 0
            }
        )
    
    # Final targeted expansion on least constrained circle with gradient-aware constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix using vectorized broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute min distances and find least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute current total sum and compute expansion direction
        current_total = np.sum(radii)
        desired_total_expansion = current_total + 0.0085  # Increase by 0.85% of the total radii
        expansion_per_circle = (desired_total_expansion - current_total) / (n - 1)
        
        # Generate expansion vector with asymmetric targeting
        # Least constrained circle gets 1.2x expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_per_circle * 1.2
        
        for i in range(n):
            if i != least_constrained_idx:
                # Apply directional expansion based on gradient-awareness
                direction_vector = centers[i] - centers[least_constrained_idx]
                norm = np.linalg.norm(direction_vector)
                if norm > 1e-10:
                    direction_vector /= norm
                expansion_factor = expansion_per_circle * 1.0 + np.random.uniform(-0.1, 0.1)
                new_radii[i] += direction_vector[0] * expansion_factor * 0.4
                new_radii[i] += direction_vector[1] * expansion_factor * 0.4
        
        # Apply constrained spatial expansion while ensuring no overlap
        expansions_done = False
        while not expansions_done:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Check for overlaps using vectorized broadcasting (with early exit)
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                expansions_done = True
            else:
                # If overlap detected, reduce total expansion by 5%
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Reoptimize final configuration with expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        res = minimize(
            neg_sum_radii, 
            v_new, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={ 
                "maxiter": 500,
                "ftol": 1e-11,
                "gtol": 1e-11,
                "eps": 1e-10,
                "disp": False,
                "iprint": 0
            }
        )
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())