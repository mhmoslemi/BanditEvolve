import numpy as np

def run_packing():
    n = 26
    # Adaptive grid refinement to balance spatial density and constraint stability
    cols = np.ceil(np.sqrt(n)).astype(int)
    rows = (n + cols - 1) // cols
    
    # Initialize with a grid-based structure with staggered rows and randomized offset
    # This improves spatial distribution while keeping initialization robust
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid positions
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add random offset to break symmetry and reduce clustering artifacts
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        
        # Stagger alternate rows for better packing geometry
        if row % 2 == 1:
            x += 0.5 / cols  # Offset alternate rows to avoid straightlining
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.32 / cols - 1e-4  # Reduced initial radius to allow more expansion potential
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3 * n total bounds

    # Objective function defined as negative of total radii to maximize
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Build constraints
    # Boundary constraints are now defined with more robust lambda closures
    cons = []
    for i in range(n):
        # Left side constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right side constraint: 1 - (x_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom side constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top side constraint: 1 - (y_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Overlap constraints: distance^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            # Use a lambda that captures the correct i and j indices at the time of definition
            # This avoids issues with late-binding variable capture
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i + 1] - v[3*j + 1])**2
                             - (v[3*i + 2] + v[3*j + 2])**2})

    # First level optimization with aggressive settings
    res = minimize(neg_sum_radii, v0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 1600, "ftol": 1e-10, "eps": 1e-8})

    # Check if optimization was successful and apply advanced perturbation strategy
    v = res.x if res.success else v0
    
    # Jiggle heuristic: randomly perturb small circles in the configuration
    # This helps the solver escape from local minima by reconfiguring less constrained areas
    # We identify smallest circles and apply small, controlled spatial shifts
    
    # Compute radii once for all operations
    if res.success:
        radii = v[2::3]
        # Track indices of all circles sorted by radius
        sorted_indices = np.argsort(radii)
        
        # Apply spatial perturbation to smaller circles
        perturbation_strength = 0.01 * (radii / radii.max())  # Proportional perturbation
        for k in range(5):  # Only alter the 5 smallest circles to avoid instability
            idx = sorted_indices[k]
            # Generate small random displacement vectors
            dx = np.random.uniform(-perturbation_strength[idx], perturbation_strength[idx])
            dy = np.random.uniform(-perturbation_strength[idx], perturbation_strength[idx])
            # Update positions
            v[3*idx] += dx
            v[3*idx + 1] += dy
        
        # Re-optimization after juggling small circles
        res = minimize(neg_sum_radii, v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 400, "ftol": 1e-10, "eps": 1e-8})
    
    # Final optimization
    if res.success:
        v = res.x
        # Compute radii again for consistency
        radii = v[2::3]
        # Compute centers
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Additional check: check all circles against boundary constraints
        # This is to ensure that perturbations didn't cause any to go out of bounds
        for i in range(n):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            if not ((x - r >= -1e-12) and (x + r <= 1 + 1e-12) and
                    (y - r >= -1e-12) and (y + r <= 1 + 1e-12)):
                # If any circle is out of bounds, rollback to last successful state
                v = res.x
                break  # Avoid checking further for efficiency
        
        # Final radius clipping to prevent numerical instability
        radii = np.clip(v[2::3], 1e-6, None)
    else:
        # If any optimization failed, fallback to last known successful state
        v = res.x
    
    return centers, radii, float(radii.sum())