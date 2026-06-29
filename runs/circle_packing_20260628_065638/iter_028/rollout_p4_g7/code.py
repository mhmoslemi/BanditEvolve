import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    initial_radius_guess = 0.30
    
    # Initialize with staggered grid and adaptive spatial jitter
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add adaptive spatial perturbation based on row spacing
        jitter_range = 0.04 * (1.0 / rows)  # smaller jitter for more dense rows
        x = x_center + np.random.uniform(-jitter_range, jitter_range)
        y = y_center + np.random.uniform(-jitter_range, jitter_range)
        
        # Staggered grid adjustment for better space utilization
        if row % 2 == 1:
            x += 0.5 / cols
        
        xs.append(x)
        ys.append(y)
    
    def perturbation_scale(radii):
        """Scale perturbations based on radius distribution to target smaller circles."""
        mean_r = np.mean(radii)
        return 0.03 * (mean_r / 0.3)  # scales perturbation proportionally with radius
    
    # Initial optimization parameters
    v0 = np.empty(3*n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, initial_radius_guess)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.45)]  # slightly tighter radius bound
    
    # Objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Boundary constraints (inequality: distance >= radius)
    cons = []
    for i in range(n):
        # Left: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right: 1 - (x_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top: 1 - (y_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Pairwise overlap constraints (ineq: distance >= r_i + r_j)
    # Use vectorized constraints for better performance
    overlap_constraints = []
    for i in range(n):
        for j in range(i+1, n):
            # Create a lambda with captures for i and j
            overlap_cons = lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            overlap_constraints.append({"type": "ineq", "fun": overlap_cons})
    
    # Optimizer setup
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=overlap_constraints + cons,
                   options={"maxiter": 3000, "ftol": 1e-10, "gtol": 1e-9})
    
    # Apply shake heuristic: Perturb smallest circles for escape from local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Select the smallest 2 circles for perturbation (based on radius)
        sorted_indices = np.argsort(radii)
        small_circle_indices = sorted_indices[:2]
        
        # Apply controlled jitter to their positions
        jitter_magnitude = 0.005  # small perturbation relative to square size
        v[3*small_circle_indices[0]] += np.random.uniform(-jitter_magnitude, jitter_magnitude)
        v[3*small_circle_indices[0] + 1] += np.random.uniform(-jitter_magnitude, jitter_magnitude)
        v[3*small_circle_indices[1]] += np.random.uniform(-jitter_magnitude, jitter_magnitude)
        v[3*small_circle_indices[1] + 1] += np.random.uniform(-jitter_magnitude, jitter_magnitude)
        
        # Re-optimize with this new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=overlap_constraints + cons,
                       options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9})
    
    # Add iterative refinement with adaptive pressure
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_matrix[i, j] = np.sqrt(dx**2 + dy**2)
                dist_matrix[j, i] = dist_matrix[i, j]
        
        # Find circles with the lowest margin between radius and current distance
        min_margins = np.zeros(n)
        for i in range(n):
            min_dist = np.min(dist_matrix[i, i+1:])
            min_margins[i] = min_dist - (radii[i] + 0.0001)  # margin with 0.0001 buffer
        
        # Select the most constrained circle for targeted optimization
        most_constrained_idx = np.argmin(min_margins)
        
        # Build a new perturbation vector to push this circle outward
        perturbation = np.zeros(3*n)
        perturbation[3*most_constrained_idx] = np.random.uniform(0.0005, 0.001)
        perturbation[3*most_constrained_idx+1] = np.random.uniform(0.0005, 0.001)
        perturbation[3*most_constrained_idx+2] = np.random.uniform(-0.0001, 0.0001)
        v = res.x + perturbation
        
        # Re-optimize with this refined configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=overlap_constraints + cons,
                       options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9})
    
    # Final cleanup and validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.45)  # clip to avoid overflow in constraints
    return centers, radii, float(radii.sum())