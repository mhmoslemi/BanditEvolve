import numpy as np

def run_packing():
    n = 26
    cols = 5  # Optimal for compact 2d packing
    rows = (n + cols - 1) // cols
    
    # Initialize seed to ensure deterministic behavior for reproducibility
    np.random.seed(42)  # Hardcoded for reproducibility and stable mutation
    
    # Create grid based on columns (x) first, then rows (y), with more spacing for topological flexibility
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add small random offset to break perfect regularity, but keep centered for stability
        # Increase random range slightly for potential topological reconfiguration, but stay <0.1
        x = x_center + np.random.uniform(-0.07, 0.07)  # Larger range for better convergence
        y = y_center + np.random.uniform(-0.09, 0.09)  # Slighter offset to allow for non-symmetrical configurations
        
        # Stagger rows for better compactness and avoid linearity
        if row % 2 == 1:
            x += 0.5 / cols * 0.9  # Slight adjustment to stagger without overlapping
        
        # Ensure we stay within 0.05 to 0.95 to avoid edge issues
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimate with improved scaling
    # Based on maximum possible radius given grid spacing and compact placement
    max_radius_estimate = (0.85 / (cols*2))  # More conservative estimate for edge effects
    # For a regular grid with spacing dx, maximum radius in square packing is ~ (dx)/2
    # So we start with a base based on grid spacing
    # Use 0.35 as baseline with smaller adjustment factor
    r0 = max_radius_estimate * 0.9  # Adjust by 10% for better initial space
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Ensure strict bounds of length 3*n, and consistent with v
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.45)]  # Radius max is 0.45 to prevent edge collisions
    
    def neg_sum_radii(v):
        """
        Objective function: maximize total radii, so we minimize the negative of it
        """
        return -np.sum(v[2::3])
    
    # Build constraints systematically with lambda-based capture and correct indexing
    # Note: use lambda with default args (i=...) to prevent closure capture issues
    # Boundary constraints: each circle has x + r <= 1.0 and x - r >= 0.0
    #                        y + r <= 1.0 and y - r >= 0.0
    # We will use vectorized approach via list comprehensions
    
    # Initialize constraints list
    cons = []
    
    # Add boundary constraints per circle (for all four sides)
    for i in range(n):
        # Left boundary: x_i - r_i >= 0.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: x_i + r_i <= 1.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y_i - r_i >= 0.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: y_i + r_i <= 1.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Add pairwise circle constraints to ensure non-overlapping using Euclidean distances
    # For each pair (i, j) where i < j, distance >= r_i + r_j
    # This is a vectorized constraint with explicit closure binding
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda to capture i and j
            def get_constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx**2 + dy**2 - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": get_constraint})
    
    # Initial optimization: more iterations with very strict tolerances
    # Use SLSQP with higher precision for gradient-based optimizations
    # Initial configuration has more room for expansion
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-11,
                                             "eps": 1e-8, "disp": False})
    
    # Post-processing: if optimization was successful, improve further
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distance matrix using vectorization for speed
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute a metric for the 'least constrained' circle
        # We use the minimum distance to other circles as a proxy
        min_distances = np.min(dists, axis=1)
        
        # Choose the circle with the largest minimum distance (least constrained)
        # This gives more freedom to expand it
        least_constrained_idx = np.argmax(min_distances)
        
        # Compute radius expansion factor with soft constraints
        current_total = np.sum(radii)
        # Targeted increase that's not too aggressive
        # Scaled by an adjustment factor to allow for better expansion without overstepping
        # 0.008 is a 0.8% increase, which is modest but enough to unlock configurations
        target_growth = 0.002  # Slight increment to trigger layout changes
        adjustment_factor = 0.8  # Soft scaling to be safer
        
        expansion_factor = (target_growth * adjustment_factor) / (n - 1)
        # Apply expansion in a way that avoids excessive over-expansion
        # Targeted expansion on least constrained circle, plus minor expansion for all
        # This helps to push the solver into better local minima
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.15  # slight extra for triggering moves
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * 1.0
            
        # Create a new decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
    
        # Re-validate and optimize again with the enhanced configuration
        # This second optimization is crucial to refine the solution
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1200, "ftol": 1e-12, "gtol": 1e-11,
                                                 "eps": 1e-8, "disp": False})
    
    # Final validation: if optimization was not successful
    v = res.x if res.success else v0
    
    # Final cleanup
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)  # Minimum radius is 1e-6 to avoid invalid values
    
    return centers, radii, float(radii.sum())