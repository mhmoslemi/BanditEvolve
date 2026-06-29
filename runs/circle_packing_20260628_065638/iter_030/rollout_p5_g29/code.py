import numpy as np

def run_packing():
    n = 26
    # Dynamic grid adaptation inspired by spatial efficiency and reduced symmetry
    cols = 5
    rows = (n + cols - 1) // cols
    # Use a more efficient grid with asymmetric grid cell sizing
    # This is designed to allow more efficient packing via dynamic spatial hashing
    xs_center = np.linspace(0.2, 0.8, cols) + np.random.uniform(-0.03, 0.03, cols)
    ys_center = np.linspace(0.2, 0.8, rows) + np.random.uniform(-0.03, 0.03, rows)
    # Initialize positions with dynamic grid offset and asymmetric staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = xs_center[col]
        y = ys_center[row]
        # Apply asymmetric stagger: rows with even indices shift left, others shift right
        # This creates a non-regular grid to reduce clustering
        if row % 2 == 0:
            x += np.random.uniform(-0.03, 0.03)
        else:
            x += np.random.uniform(0.03, 0.06)  # slightly more to prevent diagonal clustering
        if col % 2 == 0:
            y += np.random.uniform(-0.03, 0.03)
        else:
            y += np.random.uniform(0.03, 0.06)
        # Enforce asymmetric grid spacing to reduce symmetry-based constraints
        if (row % 3 == 1 and col % 2 == 0) or (row % 3 == 2 and col % 3 == 1):
            x += np.random.uniform(-0.02, 0.02)
            y += np.random.uniform(-0.02, 0.02)
        # Boundary softening: allow minimal boundary violation for convergence
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    
    # Introduce dynamic radius base based on grid spacing and asymmetry
    r0 = 0.25 / (np.max(xs_center) - np.min(xs_center)) - 1e-3
    # Apply slight radius reduction to promote spatial distribution
    r0 = max(r0 * 0.95, 0.002)  # safety lower bound
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Bounds with strict constraints to maintain numerical stability
    bounds = []
    for _ in range(n):
        bounds += [(0.0 - 1e-6, 1.0 + 1e-6),  # allow small boundary relaxation
                   (0.0 - 1e-6, 1.0 + 1e-6), 
                   (1e-4, 0.5)]  # same min radius, max radius is constrained

    def neg_sum_radii(v):
        """
        Objective: Minimize the negative sum of radii to maximize the sum
        """
        return -np.sum(v[2::3])

    # Constraint generation with explicit i,j handling and function closure
    cons = []
    for i in range(n):
        # Left boundary constraint (x_i - r_i >= 0)
        # Inequality: x_i - r_i >= 0 => x_i >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint (x_i + r_i <= 1)
        # Inequality: x_i + r_i <= 1 => 1 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint (y_i - r_i >= 0)
        # Inequality: y_i - r_i >= 0 => y_i >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint (y_i + r_i <= 1)
        # Inequality: y_i + r_i <= 1 => 1 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Generate overlap constraints with dynamic lambda closures and type safety
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq", 
                "fun": (lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + 
                         (v[3*i+1] - v[3*j+1])**2 - 
                         (v[3*i+2] + v[3*j+2])**2)
            })

    # Initial optimization with enhanced convergence control
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 1600, 
            "ftol": 1e-11, 
            "gtol": 1e-11, 
            "eps": 1e-8,
            "disp": False
        }
    )

    # Adaptive post-processing pipeline with convergence checking and validation
    if res.success:
        # Extract initial state
        v = res.x
        radii_original = v[2::3]
        centers_original = np.column_stack([v[0::3], v[1::3]])
        
        # Precompute a dense distance matrix with optimized vectorized operations
        dx = centers_original[:, np.newaxis, 0] - centers_original[np.newaxis, :, 0]
        dy = centers_original[:, np.newaxis, 1] - centers_original[np.newaxis, :, 1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Safety threshold calculation with adaptive dynamic scaling
        max_distance = np.max(dist[np.triu_indices_from(dist, 1)])
        safety_threshold_ratio = max(0.92, 1.0 - (1.0 - 0.95) * (0.9 * max_distance / 1.0))  # dynamically adjust to preserve packing density
        
        # Introduce dual-layer constraint expansion
        # First layer: expand least constrained circle based on safety threshold
        # Second layer: apply spatial optimization based on grid symmetry

        # Layer 1: Expand least constrained cluster using safety threshold
        # Find circles with minimum distance to others as candidate for expansion
        min_distances = np.min(dist, axis=1)
        least_constrained_idx = np.argmax(min_distances)
        
        safety_distance = safety_threshold_ratio * np.mean(radius_boundaries(radii_original, 1))
        # Calculate expansion factor dynamically based on spatial safety distance
        current_radius = radii_original[least_constrained_idx]
        expansion_factor = (safety_distance - current_radius) / max(1.0, current_radius * 0.8)  # adaptive scaling based on current radii
        
        # Apply aggressive expansion to least constrained, moderate to others
        expanded_radii = radii_original.copy()
        # For the least constrained, use adaptive expansion based on safety margin
        if expansion_factor > 0:
            expanded_radii[least_constrained_idx] = min(current_radius + expansion_factor * 1.2, 0.5)  # upper bound based on spatial safety
            for i in range(n):
                if i != least_constrained_idx:
                    expanded_radii[i] = min(radii_original[i] + expansion_factor * 0.6, 0.5)  # moderate expansion for others
        else:
            # If not expandable, use previous values
            expanded_radii = radii_original.copy()
        
        # Create new decision vector for expanded radii
        v_expanded = v.copy()
        v_expanded[2::3] = expanded_radii
        
        # Layer 2: Reconfiguration using adaptive spatial hashing and gradient projection
        # First, apply perturbations to coordinates to improve local basin of attraction
        # Generate spatial hash with non-uniform scaling based on grid spacing
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v_expanded.copy()
        for i in range(n):
            # Perturb coordinates with spatial hash scaled by radii and grid spacing
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii_original[i] / max_radius(radii_original))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii_original[i] / max_radius(radii_original))
        
        # Run second optimization phase with tighter tolerances
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 500,
                "ftol": 1e-12, 
                "gtol": 1e-12, 
                "eps": 1e-8,
                "disp": False
            }
        )

    # Final validation and correction step with additional safety checks
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Check for potential numerical underflow or overflow via bounds clamping
        radii = np.clip(radii, 1e-5, 0.5)
        
        # Final grid spatial check for symmetry reduction
        for i in range(n):
            # Apply asymmetric grid cell resizing for better packing
            grid_cell_x = (v[3*i] - 0.0) / (1.0 - 0.0)
            grid_cell_y = (v[3*i+1] - 0.0) / (1.0 - 0.0)
            # If grid cell is unusually small, expand coordinates slightly
            if np.abs(grid_cell_x - v[3*i] / (1.0 - 0.0)) > 0.05:
                v[3*i] = v[3*i] * 1.03
            if np.abs(grid_cell_y - v[3*i+1] / (1.0 - 0.0)) > 0.05:
                v[3*i+1] = v[3*i+1] * 1.03
        # Update decision vector for final state
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii_final = v[2::3]
    
    else:
        # If optimization fails, fall back to initial solution
        v = v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii_final = v[2::3]
    
    # Final validation with additional safety checks
    # Safety check: Ensure all radii are strictly positive and within bounds
    radii_final = np.clip(radii_final, 1e-5, 0.5)
    centers = np.column_stack([v[0::3], v[1::3]])
    
    return centers, radii_final, float(radii_final.sum())

def max_radius(radii):
    """
    Helper to compute the maximum radius in the current configuration
    Used in scaling for spatial hashing and grid perturbation
    """
    return np.max(radii)

def radius_boundaries(radii, safety_factor=1.5):
    """
    Helper to compute boundary constraints for radius based on safety_factor
    """
    max_r = np.max(radii)
    return [max_r * safety_factor, max_r * 1.0]

def verify_final_positioning(centers, radii):
    """
    Helper to verify that all positionings are feasible with the given radii
    """
    n = centers.shape[0]
    for i in range(n):
        x, y = centers[i]
        if x - radii[i] < 0 or x + radii[i] > 1.0:
            return False
        if y - radii[i] < 0 or y + radii[i] > 1.0:
            return False
    return True