import numpy as np

def run_packing():
    n = 26
    cols = 6  # Increase columns for more granular distribution 
    rows = (n + cols - 1) // cols  # Adjust rows for optimal grid
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Calculate base grid positions with refined spacing
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Use Gaussian noise for controlled variance and better diffusion
        x_noise = np.random.normal(0, 0.02)  # Reduce noise for better stability
        y_noise = np.random.normal(0, 0.02)
        
        # Add controlled offset for staggered rows (now with 2x row spacing)
        if (row % 2) == 0:
            x_center += 0.3 * (col / cols)  # More subtle staggering
        else:
            x_center -= 0.2 * (col / cols)
        
        # Final position
        x = x_center + x_noise
        y = y_center + y_noise
        
        # Add minimal jitter to avoid same position clustering
        x += np.random.uniform(-0.005, 0.005)
        y += np.random.uniform(-0.005, 0.005)
        
        xs.append(x)
        ys.append(y)
    
    # Optimal initial radius estimation based on grid dimensions and overlap constraints
    # We calculate theoretical max radius using grid edge-to-edge distance
    # Using formula (min(xspacing + yspacing)/sqrt(2)) to estimate max average
    grid_spacing_x = (1) / (cols + 1)  # Edge-to-edge with padding
    grid_spacing_y = (1) / (rows + 1)
    
    # Use 60% of minimal spacing for initial radius estimation to allow expansion room
    r0 = 0.6 * min(grid_spacing_x, grid_spacing_y) * (1 - 0.1)  # 10% safety buffer
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure 3n length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with captured i
    # Use of lambda closure with i as parameter is safe here as lambda is used correctly
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
    
    # Advanced optimization: Add spatial hash constraints for non-overlap and topological diversity
    # Use of vectorization and broadcasting for distance matrix optimization
    # This is essential for high-efficiency and precision

    # Precompute distance matrix for efficient constraint handling
    # We'll use the constraint to enforce dx^2 + dy^2 >= (r_i + r_j)^2
    # We vectorize with numpy broadcasting for performance
    # This will be handled in the constraints list

    # Vectorized overlap constraints using lambda with captured i,j
    # Added a new dimensionally-aware constraint to handle non-uniformity
    # This constraint will force minimal distance based on relative radii
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with captured i,j for constraint
            # Add a dimension-aware scaling factor for varied radius distributions
            # Use exponential scaling for tighter spatial control
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j, 
                        r_scale = np.e ** (0.1 * np.random.rand()) if np.random.rand() > 0.5 else 1.0:
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                    - r_scale * (v[3*i+2] + v[3*j+2])**2
            })

    # Initial optimization with increased max iterations and tighter tolerance
    # Added multiple phases with variable scaling to avoid convergence into flat solutions
    initial_res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                          constraints=cons, 
                          options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-9})
    
    # Phase 1: Shake and reconfigure with spatial perturbations
    if initial_res.success:
        v = initial_res.x
        
        # Compute current radius distributions
        radii = v[2::3]
        center_x = v[0::3]
        center_y = v[1::3]
        
        # Create spatial hash with adaptive weighting for enhanced reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.04  # Smaller range for better control
        
        # Apply spatial hash based on radius to ensure larger circles get more flexibility
        perturbation_scale = np.clip(radii / np.mean(radii) * 0.8, 0.2, 1.0)
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * perturbation_scale[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * perturbation_scale[i]
        
        # Re-evaluate with new spatial configuration
        phase1_res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                            constraints=cons, 
                            options={"maxiter": 450, "ftol": 1e-12, "eps": 1e-9})
        
        # Phase 2: Asymmetric radius expansion using a novel constraint-based prioritization
        if phase1_res.success:
            v = phase1_res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Compute distance matrix using broadcasting (for vectorized optimization)
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Compute minimum distance for each circle using vectorization
            min_dists = np.min(dists, axis=1)
            
            # Compute a weighted score for expansion based on: 
            # - minimum distance
            # - radius (larger circles have higher priority)
            # - distance to center
            # Normalize and compute expansion priority
            expansion_priority = (
                min_dists * (radii ** 0.8) * (0.5 + 0.8*np.sqrt( (np.min(dists**2, axis=1))/(1**2) ) )
            )
            
            # Select the most "expandable" circle based on the priority
            if np.max(expansion_priority) < 1e-12:  # all circles have zero or near-zero expansion
                isolated_idx = np.argmax(radii)  # prioritize largest circle
            else:
                isolated_idx = np.argmax(expansion_priority)
            
            # Targeted radius expansion while preserving minimal distance
            # Compute expansion factor as 1.2*base expansion + adaptive factor
            # Base expansion: (target_max + current_sum) / (n) * 0.01
            base_growth = 0.007 * (1 - (radii / np.mean(radii))**0.5)  # inverse power law
            adaptive_growth = base_growth * (1 + 0.5*(1 - np.sqrt( (radii[radii != 0] / np.mean(radii)) ) ))  # extra boost
            expansion_factor = base_growth + adaptive_growth  # total expansion factor
            
            # Create expansion vector with targeted expansion on expandable circle
            new_radii = radii.copy()
            expansion = expansion_factor * (0.95 + 0.05*np.random.rand())  # stochastic boost
            new_radii[isolated_idx] += expansion
            
            # Apply expansion with constraint validation
            # Use vectorization and gradient-safe updates for efficiency
            # We avoid full re-minimization to speed up, instead do adaptive growth
            # Apply expansion iteratively with soft constraints
            
            # We now apply a gradient-based expansion with bounds
            # Start by creating a vector
            v_new = v.copy()
            v_new[2::3] = new_radii
            
            # Re-evaluate with optimized vector and constraints
            phase2_res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                                 constraints=cons, 
                                 options={"maxiter": 350, "ftol": 1e-12, "eps": 1e-9})
            
            # Phase 3: Topology-aware perturbation with adaptive density constraints
            if phase2_res.success:
                v = phase2_res.x
                radii = v[2::3]
                centers = np.column_stack([v[0::3], v[1::3]])
                
                # Adaptive density matrix computation
                # This uses vectorization for speed
                dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
                dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
                dists = np.sqrt(dx**2 + dy**2)
                
                # Compute relative density of each circle by aggregating inverse distance to others
                density_scores = np.mean(1/(dists + 1e-12), axis=1)  # 1e-12 for numerical stability
                
                # We now create a new perturbation vector that is density sensitive
                # This uses a spatial constraint-aware perturbation
                perturbation_scales = np.clip( (density_scores / np.mean(density_scores)) * 0.8 + 0.2, 0.2, 1.0)
                
                # Generate spatial hash with more refined range (0.02-0.012)
                spatial_hash = np.random.rand(n, 2) * 0.012 * np.clip( (1 - (1.5 * (density_scores / np.mean(density_scores)) )) , 0.5, 1.0)
                
                # Apply spatial hash with density-sensitive scaling
                perturbed_v = v.copy()
                for i in range(n):
                    perturbed_v[3*i] += spatial_hash[i, 0] * (density_scores[i]) * perturbation_scales[i]
                    perturbed_v[3*i+1] += spatial_hash[i, 1] * (density_scores[i]) * perturbation_scales[i]
                
                # Re-evaluate with new spatial configuration
                phase3_res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                                     constraints=cons, 
                                     options={"maxiter": 350, "ftol": 1e-12, "eps": 1e-9})
                
                if phase3_res.success:
                    v = phase3_res.x
    
    v = initial_res.x if initial_res.success else v0
    
    # Final validation and clipping
    # We clip to ensure radii are above the tolerance level
    # Apply a final perturbation for numerical robustness
    final_perturbation = np.random.rand(n, 2) * 0.0005  # Small random jitter
    v += np.hstack([final_perturbation[:,0].reshape(-1,1), final_perturbation[:,1].reshape(-1,1), np.zeros(n).reshape(-1,1)])
    
    # Final clipping and validation to ensure constraints
    # Ensure no negative radii
    v[2::3] = np.clip(v[2::3], 1e-6, 0.5)
    
    # Ensure positions are within [0,1]
    v[0::3] = np.clip(v[0::3], 0.0, 1.0)
    v[1::3] = np.clip(v[1::3], 0.0, 1.0)
    
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    
    # Ensure final constraint validation by checking for NaNs and bounds
    # This is done via clipping and bounds in the above lines, but we validate once more
    if np.isnan(centers).any() or np.isnan(radii).any():
        # If something went wrong, fallback to the last safe version
        v = initial_res.x if initial_res.success else v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, 0.5)
    
    return centers, radii, float(radii.sum())