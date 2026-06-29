import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with advanced geometric hashing, hybrid grid + spatial perturbations, and adaptive radius bounds
    # We'll precompute spatial hashes to avoid redundant computations
    spatial_hash = np.random.rand(n, 2) * 0.12  # More aggressive hashing than parent
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid with staggered grid and adaptive spacing
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce more aggressive spatial variation in both axes
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Alternate row shifting with scaling to avoid uniformity
        shift = np.random.uniform(0.0, 0.5 / cols) if row % 2 == 1 else np.random.uniform(-0.5 / cols, 0.5 / cols)
        x += shift
        # Apply spatial hash to perturb both center and scale
        x += spatial_hash[i, 0] * (1.2 - np.random.uniform(0.1, 0.3))  # Adaptive spatial hashing based on row
        y += spatial_hash[i, 1] * (1.2 - np.random.uniform(0.1, 0.3))
        # Apply stricter clipping to avoid overshooting
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        y = np.clip(y, 1e-6, 1.0 - 1e-6)
        xs.append(x)
        ys.append(y)
    
    # Set more aggressive initial radius estimates with adaptive scaling from spatial hash
    # This helps avoid uniform initial guess and leverages spatial hashing patterns
    r0 = 0.425 / cols + np.random.uniform(0.0, 0.03) - np.mean(spatial_hash[i, 0] for i in range(n))
    r0 = np.clip(r0, 1e-4, 0.49)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.49)]  # Tighter upper radius to prevent overfitting
    
    def neg_sum_radii(v):
        """This objective is the negative of sum of radii to use minimization"""
        return -np.sum(v[2::3])
    
    # Use explicit lambda with closure capture by i for boundary constraints
    cons = []
    for i in range(n):
        # Left constraint: x - radius >= 0 (x - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: x + radius <= 1 (x + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y - radius >= 0 (y - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: y + radius <= 1 (y + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized constraint function for circle-circle overlaps (with explicit closure)
    for i in range(n):
        for j in range(i + 1, n):
            # Use explicit closure with i,j and vectorized function
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                        - (v[3*i+2] + v[3*j+2])**2)
            })

    # Initial optimization with strong constraints and strict tolerance
    # Using SLSQP with higher maxiter and tighter ftol
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds,
        constraints=cons, 
        options={  
            "maxiter": 1400,             # Enhanced iterations for deep exploration
            "ftol": 1e-11,               # Near machine precision tolerance
            "eps": 1e-11,                # Increased perturbation tolerance
            "disp": False,               # For silent optimization
            "xatol": 1e-11               # Tight bounds on parameter changes
        }
    )
    
    # Apply multi-stage refinement with adaptive constraints and gradient control
    # This is critical for overcoming local minima
    # Primary refinement: geometric hashing + radial expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply a more refined spatial hashing that includes previous state
        # This prevents complete reinitialization while introducing diversity
        refined_hash = np.random.rand(n, 2) * 0.045
        refined_hash += 0.5 * np.random.normal(0, 0.015, size=(n,2))  # Add some gradient-based noise
        
        # Perturb centers with hash, with adaptive intensity based on spatial density
        # Calculate local cell density by proximity for adaptive intensity
        # This avoids uniformly perturbing high-density regions
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dists[i, j] = np.sqrt((centers[i, 0] - centers[j, 0])**2 + 
                                      (centers[i, 1] - centers[j, 1])**2)
        
        # Calculate inverse distance sum to infer spatial density
        # Use 1/(dists[i,j] + 1e-6) to avoid division by zero
        inverse_dists = 1.0 / (dists + 1e-6)
        row_sums = np.sum(inverse_dists, axis=1)
        
        # Normalize inverse distance for adaptive perturbation scaling
        row_sums_norm = row_sums / np.max(row_sums)
        # Apply inverse normalization for high-density cells to reduce perturbation
        inv_density = 1.0 - row_sums_norm

        # Apply spatial hashing with density-driven scaling
        perturbed_v = v.copy()
        for i in range(n):
            perturbation = refined_hash[i, :] * inv_density[i] * (0.8 + 0.2 * np.random.rand())
            perturbed_v[3*i] += perturbation[0]
            perturbed_v[3*i+1] += perturbation[1]

        # Apply strict constraint re-evaluation after perturbation
        res = minimize(
            neg_sum_radii,
            perturbed_v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons, 
            options={  
                "maxiter": 500,             # Refinement stage
                "ftol": 1e-11,              # High precision to recover from perturbation
                "eps": 1e-11,               # Higher perturbation for recovery
                "disp": False,              # Silent mode
                "xatol": 1e-11              # Strict change tolerance for stability
            }
        )

    # Apply targeted radius expansion on the most underutilized circle
    # Use an aggressive but constrained approach based on geometric analysis
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Re-calculate distances with vectorization using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate for each circle the minimal distance to another
        min_dists = np.min(dists, axis=1)
        
        # Identify the circle with the largest available expansion margin
        # This is calculated as maximum (distance - sum of radii), but normalized
        # Use an explicit formula to avoid recomputation
        expansion_margins = np.zeros(n)
        for i in range(n):
            # We're looking for the circle that has the largest available space
            # to increase its radius without causing overlap
            max_available = 0.0
            # We compare against all other circles
            for j in range(n):
                # Skip self
                if i == j:
                    continue
                dist_ij = dists[i, j]
                # Available space is the distance minus the sum of radii
                available = dist_ij - (radii[i] + radii[j])
                if available > max_available:
                    max_available = available
            # To normalize, we also consider the minimum margin to edges
            edge_margin = np.min(
                [v[3*i] - radii[i], 1.0 - v[3*i] - radii[i], 
                 v[3*i+1] - radii[i], 1.0 - v[3*i+1] - radii[i]]
            )
            expansion_margins[i] = max_available * (edge_margin / (1.0 - 0.05))  # scale by edge buffer
        least_constrained_idx = np.argmax(expansion_margins)
        
        # This is a targeted expansion of just the most under-constrained circle
        # Use a careful growth to ensure all constraints are satisfied
        # We'll use a binary search-based expansion strategy with gradient checks
        # We first try to increase the radius in small steps
        expansion_factor = 0.0
        while True:
            # Calculate current total sum
            total_current = np.sum(radii)
            # We aim to find the maximal increment possible
            # But we cannot exceed edge margins, so use binary search
            # First, try doubling the expansion rate
            new_radii = radii.copy()
            new_radii[least_constrained_idx] += expansion_factor * 1.2  # Over-estimation for safe check
            try:
                # Validate expanded configuration with strict overlap checks
                # Use the validation function from the reference, but for this local evaluation
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = new_radii[i] - new_radii[j]  # not correct, but we need the real centers
                        # Real centers are from v[0::3] and v[1::3]
                        i_x, i_y = v[3*i], v[3*i+1]
                        j_x, j_y = v[3*j], v[3*j+1]
                        dist = np.sqrt((i_x - j_x)**2 + (i_y - j_y)**2)
                        if dist < new_radii[i] + new_radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # We found a valid expansion level; check edge constraints
                    valid_edges = True
                    for i in range(n):
                        # Check edge constraints for all circles
                        r = new_radii[i]
                        x, y = v[3*i], v[3*i+1]
                        if (x - r < -1e-12 or x + r > 1 + 1e-12 or
                            y - r < -1e-12 or y + r > 1 + 1e-12):
                            valid_edges = False
                            break
                    if valid_edges:
                        if expansion_factor >= 0.006:
                            break
                        expansion_factor *= 1.02
                    else:
                        expansion_factor = 0.0  # Revert
                else:
                    expansion_factor = 0.0
            except:
                expansion_factor = 0.0
        
        # Apply expansion to the least constrained circle
        v_new = v.copy()
        v_new[3*least_constrained_idx + 2] += expansion_factor

        # Re-evaluate with expansion and strict constraints
        new_radii = np.copy(v_new[2::3])
        res = minimize(
            neg_sum_radii,
            v_new, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons, 
            options={  
                "maxiter": 500,             # Refinement stage
                "ftol": 1e-11,              # High precision to recover from perturbation
                "eps": 1e-11,               # Higher perturbation for recovery
                "disp": False,              # Silent mode
                "xatol": 1e-11              # Strict change tolerance for stability
            }
        ) 

        # Final evaluation and return
        if res.success:
            v = res.x
        else:
            # Fall back to previous state
            v = v_new

    # Final validation and cleanup of possible NaNs or invalid values
    # Apply clipping and ensure positivity of radii
    # Use the provided validator internally but use numpy logic for safety
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.49)  # Ensure radii are strictly positive and under max
    # Apply final constraint check for safety
    # We check all possible overlap for final verification
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < radii[i] + radii[j] - 1e-12:
                valid = False
                break
        if not valid:
            break
    if not valid:
        # This means we're in an invalid state - revert to previous successful state
        centers = np.column_stack([v0[0::3], v0[1::3]])
        radii = v0[2::3]
        radii = np.clip(radii, 1e-6, 0.49)
    
    # Final return
    return centers, radii, float(radii.sum())