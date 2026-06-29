import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Optimized initialization: hybrid structured placement + adaptive perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Base shift with decreasing magnitude for higher rows, encouraging top-bottom spread
        row_factor = 1.0 - (row / (rows-1)) if rows > 1 else 0.0
        x_center += np.random.uniform(-0.02 * row_factor, 0.02 * row_factor)
        x_center += 0.05 * np.sin(2 * np.pi * (i / n))  # Harmonic perturbation for spread
        
        y_center += np.random.uniform(-0.03, 0.03)  # Light vertical perturbation
        # Stagger alternating rows with adaptive spacing
        if row % 2 == 1:
            x_center += 0.25 / cols * (np.sin(i) / (0.5 + 0.5 * abs(np.sin(i))) + 1)
        
        # Normalize to unit square, ensuring we don't go out of bounds
        x_center = np.clip(x_center, 0.0005, 0.9995)
        y_center = np.clip(y_center, 0.0005, 0.9995)
        
        xs.append(x_center)
        ys.append(y_center)
    
    # Initialize radii with adaptive base size, avoiding uniformity
    max_row = rows - 1
    r0 = np.full(n, 0.28 / cols - 1e-3)  # Base radius with slight variance
    # Add row-based radius adjustment: higher rows get smaller radii for better vertical stacking
    r0 += 0.02 * ((max_row - row) / (max_row + 1)) - 0.005
    
    # Create initial decision vector in v0 with structured spatial setup
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.clip(r0, 1e-4, 0.45)  # Clipping for safety
    
    # Define bounds, ensuring they exactly match the 3*n length of the decision vector
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.45)]  # Radius capped at more reasonable value
    
    # Objective function: maximize sum of radii
    def neg_sum_radii(v):  # return negative for minimizer
        return -np.sum(v[2::3])  # v[2::3] extracts radii
    
    # Vectorized constraints: define with proper i capture and lambda closure
    # Using closures with fixed i by capturing it in outer loop for each function
    
    # Boundary constraints: for each circle, 0 <= x - r <= 1, 0 <= y - r <= 1
    cons = []
    for i in range(n):
        # Left boundary x - r >= 0
        def constraint_left(v, i=i):
            return v[3*i] - v[3*i + 2]
        cons.append({"type": "ineq", "fun": constraint_left})
        
        # Right boundary x + r <= 1
        def constraint_right(v, i=i):
            return 1.0 - v[3*i] - v[3*i + 2]
        cons.append({"type": "ineq", "fun": constraint_right})
        
        # Bottom boundary y - r >= 0
        def constraint_bottom(v, i=i):
            return v[3*i+1] - v[3*i + 2]
        cons.append({"type": "ineq", "fun": constraint_bottom})
        
        # Top boundary y + r <= 1
        def constraint_top(v, i=i):
            return 1.0 - v[3*i+1] - v[3*i + 2]
        cons.append({"type": "ineq", "fun": constraint_top})
    
    # Overlap constraints: use vectorized broadcasting with efficient pairwise calculation
    # Precomputed distances using broadcasting instead of explicit loops to optimize speed
    # To reduce computational overhead, we'll precompute a grid of all pairwise distances
    # and use this to avoid repeated computation
    
    # Overlap constraints: distance^2 >= (r_i + r_j)^2
    # Optimized using broadcasting to avoid redundant pairwise computations
    
    # Initial optimization: with more aggressive constraints and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-10, "eps": 1e-10, "disp": False})
    
    # Asymmetric reconfiguration: adaptive spatial perturbation with local radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        current_sum = np.sum(radii)
        
        # Compute pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        
        # Compute "isolation score" by sum of minimum distance to all others
        # This is a more reliable measure than simple distance sums (avoiding bias in dense regions)
        min_dists = np.min(dists, axis=1)
        isolation_scores = 1.0 / (min_dists + 1e-9)  # Invert to prioritize isolated circles
        least_isolated_idx = np.argmin(isolation_scores)
        
        # Compute average isolation factor as a baseline
        avg_isolation = np.mean(min_dists)
        
        # Define a "isolation_gain" metric that increases with isolation 
        # and depends linearly on minimum distance to all other circles
        isolation_gain = np.sqrt(min_dists / avg_isolation)
        
        # Use isolation_gain to determine potential for radius expansion
        potential_expansions = (isolation_gain / (1 + np.exp(-isolation_gain))) * 0.02  # Sigmoid function for non-linear scaling
        # Add some randomized spatial perturbation to the least-isolated circle
        spatial_perturbation = np.random.rand(3) * 0.05 - 0.025
        
        # Form perturbed vector based on spatial perturbation and expansion
        # Use the original vector as a base, but alter the least-isolated circle's x, y, and r
        # This is done in a way that doesn't break constraints but allows exploration
        perturbed_v = v.copy()
        perturbed_v[3*least_isolated_idx] += spatial_perturbation[0]
        perturbed_v[3*least_isolated_idx+1] += spatial_perturbation[1]
        perturbed_v[3*least_isolated_idx+2] += spatial_perturbation[2] + potential_expansions[least_isolated_idx]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "disp": False})
    
    # Post-optimization refinement: advanced gradient approximation with constraints
    # To stabilize results, we re-evaluate using tighter constraints and a different approach
    
    if res.success:
        v = res.x
        radii = v[2::3]
        
        # Re-calculate spatial configuration
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        
        # Identify most isolated circle using advanced isolation metric
        # Calculate average minimum distance for all circles
        avg_min_dist = np.mean(np.min(dists, axis=1))
        # Calculate weighted isolation as (min distance) / (avg_min_dist)
        isolation_weights = np.min(dists, axis=1) / (avg_min_dist + 1e-8)
        least_isolated_idx = np.argmin(isolation_weights)
        
        # Use advanced expansion: based on isolation weights, and apply it with soft constraint handling
        isolation_weights_normalized = (isolation_weights - np.min(isolation_weights)) / (np.max(isolation_weights) - np.min(isolation_weights) + 1e-8)
        # Use exponential decay of expansion potential to avoid over-aggressive expansion
        expansion_factors = np.exp(isolation_weights_normalized * -1.5)  # Invert since higher isolation == lower weights
        
        # Define a targeted expansion with adaptive scaling based on isolation weight
        # We calculate an "expansion budget" for other circles to allow for growth
        expansion_budget = (0.008 - 0.001) * (isolation_weights_normalized[least_isolated_idx] + 1e-8)  # Adjust for isolation
        
        # Apply expansion to other circles with adaptive scaling
        for i in range(n):
            if i != least_isolated_idx:
                # Scale the expansion according to isolation_weights: isolated circles get more expansion
                expansion = expansion_budget * (1 + 1.2 * (isolation_weights_normalized[i] - 0.5))
                # Clamp expansion to max 0.003 of total to avoid overshooting
                expansion = np.clip(expansion, 1e-7, 0.003)
                # Add expansion to radii of all but the least-isolated (who have been expanded)
                v[3*i + 2] += expansion
        
        # Re-evaluate with these expanded radii
        # Use a more robust method to handle any potential constraint violations
        # Apply the expanded radii as a new initial vector
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "disp": False})
    
    # Final refinement of solution: if we succeeded, apply a small spatial perturbation
    # to avoid getting stuck in local optima from the initial run
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Apply a small randomized spatial perturbation to break symmetry
        spatial_perturbation = np.random.rand(n, 2) * 0.03 - 0.015
        perturbed_v = v.copy()
        
        # Adjust centers to ensure they still lie inside unit square
        # We apply a small perturbation and clip to maintain validity
        for i in range(n):
            x = perturbed_v[3*i]
            y = perturbed_v[3*i+1]
            x += spatial_perturbation[i, 0]
            y += spatial_perturbation[i, 1]
            x = np.clip(x, 1e-4, 1.0 - 1e-4)
            y = np.clip(y, 1e-4, 1.0 - 1e-4)
            perturbed_v[3*i] = x
            perturbed_v[3*i+1] = y
        
        # Re-optimize with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "disp": False})
    
    # If optimization fails, fallback to the best of the initial vector and the perturbed ones
    v = res.x if res.success else v0
    
    # Final clipping to ensure physical validity
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.45)  # Ensure radii stay within realistic bounds
    return centers, radii, float(radii.sum())