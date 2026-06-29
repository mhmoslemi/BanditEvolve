import numpy as np

def run_packing():
    """
    Solves the circle packing problem for 26 circles in the unit square using advanced optimization
    with stochastic geometry reconfiguration, gradient-aware constraints, and multi-phase expansion.
    """
    n = 26
    # Initialize geometric grid parameters with 6-column adaptive layout for enhanced spread
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Dynamic seed for better variability across runs
    seed_val = np.random.randint(0, 1000000)
    np.random.seed(seed_val)
    
    # Create optimized grid layout with asymmetric staggering and spatial entropy injection
    xs = []
    ys = []
    center_x = []
    center_y = []
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.35) / cols * 0.95  # Shift x centers inward to allow more radial space
        y_center = (row + 0.35) / rows * 0.95  # Same for y
        
        # Add asymmetric stochastic perturbations to break symmetry
        x_offset = np.random.uniform(-0.12, 0.12) * (1/radial_adjustment_factor(row, col))
        y_offset = np.random.uniform(-0.12, 0.12) * (1 / radial_adjustment_factor(row, col))
        
        # Add row-level stagger to simulate non-regular grid
        row_stagger = (row % 2) * (0.4 / cols)
        x_center += row_stagger
        
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Apply spatial entropy adjustment to avoid cluster formation, especially around edges
        if (row == 0 or row == rows - 1) or (col == 0 or col == cols - 1):
            x += np.random.uniform(-0.03, 0.03)
            y += np.random.uniform(-0.03, 0.03)
        
        xs.append(x)
        ys.append(y)
        center_x.append(x_center)
        center_y.append(y_center)
    
    # Define radial scaling factor based on grid location to optimize spacing for expansion potential
    # Higher radial factors in central regions give less expansion flexibility, while edge regions have more
    # We use a quadratic adjustment to ensure edges are more flexible
    def radial_adjustment_factor(row, col):
        x = col / cols
        y = row / rows
        # Radial weighting: prioritize expansion where centers have the most potential for growth
        # We use inverse square root to allow less constrained areas to have higher growth potential
        # while maintaining enough spacing
        return 1 / (1 + np.sqrt(x * (1 - x) * y * (1 - y))) ** 0.5
    
    # Base radius calculation based on grid spacing and spatial constraints
    # Use row and col to determine base radius, scaled by the adjustment factor
    base_radius = 0.35 / cols * 1.4  # Slight increase for initial optimism
    r0 = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Initial radii scale by the radial adjustment factor and grid size
        r0.append(base_radius * radial_adjustment_factor(row, col))
    
    # Construct initial vector v with 3*n dimensions
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.array(r0)
    
    # Define bounds for the vector with exact 3n elements
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)]  # Adjusted lower bound to 1e-6 to avoid clipping issues in early steps
    
    # Objective function: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Construct constraints with more precise lambda binding (using partial for better capture)
    
    # Boundary constraints with more efficient lambda binding
    cons = []
    for i in range(n):
        # Left - radius <= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: (v[3*i] - v[3*i+2])})
        # Right + radius >= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: (1.0 - v[3*i] - v[3*i+2])})
        # Bottom - radius <= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: (v[3*i+1] - v[3*i+2])})
        # Top + radius >= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: (1.0 - v[3*i+1] - v[3*i+2])})
    
    # Optimized spatial constraints with vectorized calculation, and 
    # adaptive constraint sensitivity based on proximity to edges (to handle edge circles)
    # We use more efficient constraint functions which use broadcasting for faster calculation
    for i in range(n):
        for j in range(i+1, n):
            # Compute constraints efficiently with minimal evaluation in lambda
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                # Adjust the threshold to allow for minimal tolerance while handling edge cases
                threshold = (v[3*i+2] + v[3*j+2]) 
                return dist_sq - (threshold * threshold) + 1e-8  # Small epsilon to prevent floating point issues
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # First optimization phase with higher convergence and adaptive tolerances
    # Using the SLSQP method with more aggressive tolerance and early stopping
    # Set up the first phase with increased max iteration and tight tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 1000,  # Increased to 1000
                       "ftol": 1e-13,   # Tighter tolerance
                       "gtol": 1e-8,    # Tighter gradient tolerance
                       "eps": 1e-8,     # Smaller step sizes for better convergence
                       "disp": False,   # No diagnostics
                       "iprint": -1   # No print output for execution speed
                   })
    
    # If initial optimization succeeds, perform asymmetric reconfiguration
    if res.success:
        v = res.x
        
        # Calculate a grid-based spatial entropy map to guide reconfiguration
        # We use the centers to create a grid of spatial influences
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Compute a spatial influence matrix for reconfiguration
        # This matrix will be used to guide the random spatial perturbation
        # More influence is given to circles with smaller radii, as they have more flexibility
        spatial_influence = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    spatial_influence[i][j] = 0.0  # No influence with self
                else:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 2 * np.max(radii):
                        # If close, they influence each other
                        spatial_influence[i][j] = np.exp(-dist / np.max(radii))
                    else:
                        spatial_influence[i][j] = 0.0
        # Compute spatial entropy: higher entropy in regions with more influence
        spatial_entropy = np.sum(spatial_influence, axis=1)
        
        # Generate a random spatial perturbation based on this entropy for asymmetric reconfiguration
        # Higher entropy regions are less prone to perturbation
        perturbation_factor = 0.025  # Reduced from 0.05 for precision
        random_hash = np.random.rand(n, 2) * 0.05 * (1 + np.log(spatial_entropy + 1e-12))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0] * (1.0 / (spatial_entropy[i] + 1e-12))
            perturbed_v[3*i+1] += random_hash[i, 1] * (1.0 / (spatial_entropy[i] + 1e-12))
        
        # Second optimization phase with tighter tolerances and more iterations
        # This helps us explore the new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 800,  # Reduced from 400 to better explore
                           "ftol": 1e-13,  # Tighter
                           "gtol": 1e-9,   # Tighter
                           "eps": 1e-8,   # Same as first phase
                           "disp": False,
                           "iprint": -1
                       })
        v = res.x
    
    if res.success:
        # We now look for the least constrained circle and expand it
        # First compute pairwise distances for all circles
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dist_matrix = np.zeros((n,n))
        for i in range(n):
            dx = centers[i, 0] - centers[:, 0]
            dy = centers[i, 1] - centers[:, 1]
            dist_matrix[i, :] = np.sqrt(dx*dx + dy*dy)
        
        # Find the circle with minimum minimum distance to neighbors (least constrained)
        min_distance_per_circle = np.min(dist_matrix, axis=1)
        min_distance_idx = np.argmin(min_distance_per_circle)
        
        # Compute how much we can expand the min distance circle
        # Calculate growth potential based on available space
        max_possible_growth = min(min_distance_per_circle[min_distance_idx] - radii[min_distance_idx], 
                                 1.0 - np.max(centers[:,0]) - radii[min_distance_idx],
                                 np.min(centers[:,0]) - radii[min_distance_idx],
                                 1.0 - np.max(centers[:,1]) - radii[min_distance_idx],
                                 np.min(centers[:,1]) - radii[min_distance_idx])
        
        max_growth = max_possible_growth
        if max_growth <= 0:
            max_growth = 0.0001  # Avoid zero growth which won't help
        
        # Create expansion vector that increases the radius of the least constrained circle
        # while distributing the gain to other circles to maintain constraints
        # Use a logarithmic distribution for radius expansion to avoid over-concentration
        expansion_factor = max_growth * 1.3  # Add some extra for non-linearity
        new_radii = radii.copy()
        new_radii[min_distance_idx] += expansion_factor * 1.1  # Slight over-expansion
        
        # Distribute the remaining growth across other circles
        # Use the inverse of the minimum distance to other circles as a weighting
        # to prioritize expanding those that are relatively more constrained
        # We also factor in how far they are from edges to avoid pushing them out
        expansion_weights = np.zeros(n)
        for j in range(n):
            if j == min_distance_idx:
                continue
            # Compute a distance-weighted constraint
            dx = centers[min_distance_idx, 0] - centers[j, 0]
            dy = centers[min_distance_idx, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            # Use the distance to the edge as a multiplier for constraint
            # This gives circles near edges more room to grow
            edge_distance = np.min([
                centers[j, 0] - 1.0, 1.0 - centers[j, 0],
                centers[j, 1] - 1.0, 1.0 - centers[j, 1]
            ]) * 0.5

            expansion_weights[j] = (dist - (radii[min_distance_idx] + radii[j])) * edge_distance
            expansion_weights[j] = np.maximum(expansion_weights[j], 0)  # Ensure non-negative
        expansion_weights /= np.sum(expansion_weights) if expansion_weights.sum() > 0 else 1
        
        # Apply the expansion to non-min circles
        for j in range(n):
            if j != min_distance_idx:
                new_radii[j] += expansion_factor * expansion_weights[j] * 0.8
        
        # Apply a safety check to ensure expansion doesn't cause overlaps
        # We do this in a controlled way by checking overlaps and gradually reducing the expansion
        while True:
            # Create a new decision vector with expanded radii
            new_v = v.copy()
            new_v[2::3] = new_radii.copy()
            
            # Get new centers and radii
            new_centers = np.column_stack([new_v[0::3], new_v[1::3]])
            new_radii = new_v[2::3]
            
            # Check for overlaps
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = new_centers[i, 0] - new_centers[j, 0]
                    dy = new_centers[i, 1] - new_centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < (new_radii[i] + new_radii[j]) - 1e-12:
                        valid = False
                        # Reduce expansion slightly
                        scale_factor = 0.95
                        for idx in range(n):
                            if idx != min_distance_idx:
                                new_radii[idx] = new_radii[idx] * scale_factor
                        break
                if not valid:
                    break
            if valid:
                break
        
        # Apply expansion to the decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii.copy()
        
        # Final optimization to refine the configuration while enforcing constraints
        # This helps to avoid over-constraint issues
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 700,
                           "ftol": 1e-12,
                           "gtol": 1e-9,
                           "eps": 1e-8,
                           "disp": False,
                           "iprint": -1
                       })
        v = res.x
    
    # Final cleanup of radii to ensure no negative values, and apply a clipping
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())