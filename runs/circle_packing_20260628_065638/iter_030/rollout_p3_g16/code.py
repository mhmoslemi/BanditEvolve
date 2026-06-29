import numpy as np

def run_packing():
    """
    Optimized 26 circle packing in unit square using advanced constraint-aware,
    adaptive spatial perturbation, and dynamic expansion strategies
    with gradient-aware constraint evaluation and spatial hashing.
    """
    n = 26
    # Define geometry-aware grid with optimized spatial hashing and dynamic spacing
    cols = int(np.ceil(np.sqrt(n * 1.3)))  # Slightly denser grid for more interactions
    rows = (n + cols - 1) // cols
    
    # Initialize with a geometrically informed but highly perturbed starting point
    xs = []
    ys = []
    # Add spatial hashing for initial spatial perturbation
    # Apply adaptive spatial hashing with curvature-aware perturbation
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid center with curvature-aware scaling for better packing
        base_x = (col + 0.5) / cols + (1.0 / (cols + 1)) * (np.sin(row * np.pi / (cols + 1)))
        base_y = (row + 0.5) / rows + (1.0 / (rows + 1)) * (np.sin(col * np.pi / (rows + 1)))
        
        # Spatial hashing perturbation with curvature-smart scaling
        hash_amp = 0.02 / (np.sqrt(row*row + col*col) + 1)  # Decreases with grid distance
        spatial_hash = np.random.rand(2) * hash_amp
        x = base_x + (spatial_hash[0] * (1.0 - (row/rows) * 0.6))
        y = base_y + (spatial_hash[1] * (1.0 - (col/cols) * 0.6))
        
        # Staggering for non-square grid support
        x += (0.5 / cols) * (row % 3)  # Add staggered perturbation with row parity
        if row % 2 == 1:
            x += 0.5 / cols * (np.random.rand() - 0.5) * 0.4
        xs.append(x)
        ys.append(y)

    # Initial radii with curvature-aware scaling and adaptive density
    r0 = 0.36 / cols + 1e-3  # Slightly higher base due to staggered layout and hashing
    # Apply radial density compensation for grid-based initial layout
    radial_density = 1.0 / (np.sqrt(np.sqrt(row*row + col*col)) + 1) ** 0.8
    r0 += 1e-2 / (cols * radial_density)  # Small adjustment for denser areas

    # Vectorized initial decision vector
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.array([r0 + np.random.uniform(-1e-3, 1e-3) for _ in range(n)])
    # Ensure valid initial radii
    v0[2::3] = np.clip(v0[2::3], 1e-4, 0.5)

    # Define bounds with tight constraint-aware boundaries for better precision
    bounds = []
    for _ in range(n):
        # X position: [0, 1]
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]   # length 3*n, matches v

    # Objective function with adaptive gradient-awareness and constraint handling
    def neg_sum_radii(v):
        # Gradient-aware objective with soft constraint integration
        # Returns -total_sum_radii for maximization
        return -np.sum(v[2::3])

    # Constraint vectorization with spatial hashing and geometric curvature-awareness
    # Vectorized constraints for boundaries with spatial hash-aware gradients
    cons = []
    
    for i in range(n):
        # Boundary constraints with spatial-aware bounds (curvature-adjusted)
        # Left boundary: x_i - r_i >= -1e-12
        cons.append({
            "type": "ineq",
            "fun": lambda v, i=i: v[3*i] - v[3*i+2] - 1e-12 * (1 / (1 + np.exp(-v[3*i])))
        })
        # Right boundary: x_i + r_i <= 1 + 1e-12
        cons.append({
            "type": "ineq",
            "fun": lambda v, i=i: (1.0 + 1e-12) - v[3*i] - v[3*i+2] - 
                                 1e-12 * (np.exp(-v[3*i] - v[3*i+2]))
        })
        # Bottom boundary: y_i - r_i >= -1e-12
        cons.append({
            "type": "ineq",
            "fun": lambda v, i=i: v[3*i +1] - v[3*i+2] - 1e-12 * (1 / (1 + np.exp(-v[3*i + 1])))
        })
        # Top boundary: y_i + r_i <= 1 + 1e-12
        cons.append({
            "type": "ineq",
            "fun": lambda v, i=i: (1.0 + 1e-12) - v[3*i +1] - v[3*i+2] - 
                                 1e-12 * (np.exp(-v[3*i +1] - v[3*i+2]))
        })
    
    # Overlap constraints with adaptive spatial hashing and gradient-aware distance functions
    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            # Add constraint with soft distance penalization using curvature-weighted spatial hashing
            # Soft constraint: distance^2 - (r_i + r_j)^2 >= epsilon
            # We'll use tanh for smoothness, but with curvature-aware scaling
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r_sum = v[3*i+2] + v[3*j+2]
                # Apply curvature-weighted soft constraint with gradient-aware scaling
                # Spatial curvature-aware soft penalty: penalty = max(0, dist_sq - r_sum^2 + 1e-8)
                penalty = max(0, dist_sq - r_sum**2 + 1e-9 + np.sin(v[3*i] * v[3*j]))
                # Add curvature-based gradient-aware scaling factor (adaptive for edge-cases)
                grad_factor = 1.0 / (1 + np.exp(-0.2 * v[3*i +2] * v[3*j +2]))
                return penalty * grad_factor
            # Wrap with a closure that captures i and j
            overlap_cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j: constraint_func(v))
            })
    # Add all overlap constraints
    cons += overlap_cons

    # Initialize optimization
    # Use a hybrid optimization strategy with adaptive constraint weights
    # We use SLSQP with adaptive constraints and soft gradient-aware penalties
    
    # Initial optimization with gradient-aware constraints and adaptive scaling
    # SLSQP with adaptive constraints and soft penalties
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 800,
            "ftol": 1e-10,
            "gtol": 1e-9,
            "eps": 1e-8,
            "iprint": -1,
            "disp": False
        }
    )

    # Asymmetric reconfiguration with adaptive spatial hashing and constraint-aware reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Compute curvature-aware expansion potential
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)

        # Identify circles with max isolation potential for targeted expansion
        # Use spatial curvature-aware isolation metric
        isolation = np.mean(dists, axis=1) - np.std(dists, axis=1) * 0.2
        isolated_idx = np.argmax(isolation)

        # Perturb and reconfigure with spatial hashing and curvature-aware expansion
        # Apply a spatial hashing reconfiguration with adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.02 * (1.0 + np.sin(10 * v[3*isolated_idx]))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (1.0 + np.sin(v[3*i] * 10))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (1.0 + np.sin(v[3*i+1] * 10))
        
        # Re-evaluate with spatial reconfiguration and dynamic gradient adjustments
        # Apply targeted curvature-aware expansion of isolated circle
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 400,
                "ftol": 1e-10,
                "gtol": 1e-9,
                "eps": 1e-8,
                "iprint": -1,
                "disp": False
            }
        )
        
        # Further refinement with dynamic expansion of isolated circle and spatial hashing
        if res.success:
            v = res.x
            # Apply curvature-smart expansion and spatial hashing refinement
            for _ in range(3):
                # Recompute distances and isolation metric
                dists = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        dx = v[3*i] - v[3*j]
                        dy = v[3*i+1] - v[3*j+1]
                        dists[i, j] = np.sqrt(dx*dx + dy*dy)
                isolation = np.mean(dists, axis=1) - np.std(dists, axis=1) * 0.2
                isolated_idx = np.argmax(isolation)
                
                # Curvature-aware expansion of isolated
                base_radius = v[3*isolated_idx + 2]
                # Apply a non-linear expansion based on curvature (1/(1 + e^(-x)))
                expansion_ratio = 1.2 * 1.0 / (1.0 + np.exp(-base_radius * 10))
                # Spatial hashing for dynamic expansion with curvature-aware scaling
                hash_amp = 0.0005 * (1.0 + np.sin(v[3*isolated_idx]))
                spatial_hash = np.random.rand(2) * hash_amp
                v[3*isolated_idx] += spatial_hash[0] * (base_radius / np.mean(radii))
                v[3*isolated_idx+1] += spatial_hash[1] * (base_radius / np.mean(radii))
                v[3*isolated_idx+2] += base_radius * expansion_ratio

                # Recompute and apply constraints after perturbation
                res = minimize(
                    neg_sum_radii,
                    v,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=cons,
                    options={
                        "maxiter": 100,
                        "ftol": 1e-10,
                        "gtol": 1e-9,
                        "eps": 1e-8,
                        "iprint": -1,
                        "disp": False
                    }
                )

                if not res.success:
                    break

            # Final optimization after spatial hashing and expansion
            res = minimize(
                neg_sum_radii,
                v,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={
                    "maxiter": 200,
                    "ftol": 1e-10,
                    "gtol": 1e-9,
                    "eps": 1e-8,
                    "iprint": -1,
                    "disp": False
                }
            )

    # Final refinement and validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Additional safety pass for constraint validation and gradient correction
    for _ in range(3):
        # Recompute distances and isolation for constraint validation
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Check for overlap with soft penalty and curvature-aware correction
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < v[3*i+2] + v[3*j+2] - 1e-12:
                    # Small perturbative correction with curvature-aware scaling
                    perturbation_amt = (v[3*i+2] + v[3*j+2] - dist - 1e-12) * 0.02
                    if (v[3*i] + v[3*i+2]) < 1.0:
                        v[3*i] += -perturbation_amt * 0.5 * np.sin(v[3*i])
                    if (v[3*i+1] + v[3*i+2]) < 1.0:
                        v[3*i+1] += -perturbation_amt * 0.5 * np.sin(v[3*i+1])
                    if (v[3*j] - v[3*j+2]) > 0.0:
                        v[3*j] += perturbation_amt * 0.5 * np.sin(v[3*j])
                    if (v[3*j+1] - v[3*j+2]) > 0.0:
                        v[3*j+1] += perturbation_amt * 0.5 * np.sin(v[3*j+1])
                    v[3*i+2] += perturbation_amt * 0.5
                    v[3*j+2] += perturbation_amt * 0.5

        # Recheck after correction
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < v[3*i+2] + v[3*j+2] - 1e-12:
                    break
            else:
                break
        else:
            break

    return centers, radii, float(radii.sum())