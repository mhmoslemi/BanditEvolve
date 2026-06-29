import numpy as np

def run_packing():
    n = 26
    # Optimal column arrangement with dynamic row allocation
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with geometric hashing and perturbation
    # First, define an adaptive initial spacing based on expected radii
    # We'll use higher initial spacing to allow room for expansion
    base_radius = 0.45 / cols
    spacing_factor = 1.15  # allow for better radial expansion
    
    # Define an initial "spatial fingerprint" to break symmetry
    # Use a grid-like but perturbed arrangement with dynamic offset
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base position on a staggered grid
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Apply spatial hashing to avoid direct grid clustering
        hash_x = np.sin(np.random.rand() * 20) * 0.05
        hash_y = np.cos(np.random.rand() * 20) * 0.05
        x_offset = (1 / cols) * np.sin(row * 2 * np.pi / rows) * 0.04
        y_offset = (1 / rows) * np.cos(col * 2 * np.pi / cols) * 0.04
        
        x = base_x + x_offset + hash_x
        y = base_y + y_offset + hash_y
        
        # Ensure we don't exceed unit square
        x = max(min(x, 1.0 - base_radius * spacing_factor), base_radius * spacing_factor)
        y = max(min(y, 1.0 - base_radius * spacing_factor), base_radius * spacing_factor)
        
        # Store in list
        xs.append(x)
        ys.append(y)
    
    # Initial radii: based on expected spacing
    base_radius = 0.27 / cols
    r0 = base_radius * spacing_factor - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for the 3n variables
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function: maximize sum of radii (minimize negative sum)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Generate constraints using vectorized approach with proper capture
    cons = []
    
    # Boundary constraints per circle
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints between every pair of circles
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with closure to ensure correct i and j for each constraint
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                        - (v[3*i+2] + v[3*j+2])**2)
            })

    # First optimization with adaptive scaling and tight tolerances
    # Use SLSQP with tighter ftol and more steps with warm starting strategy
    res = minimize(neg_sum_radii, v0, 
                   method="SLSQP", bounds=bounds,
                   constraints=cons,
                   options={"maxiter": 1200, "ftol": 1e-12, "eps": 1e-3})

    # Introduce "dynamic recomposition phase": analyze top 2 interacting circles
    # Identify top two closest pairs of circles (in terms of distance between centers)
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.zeros(n*n)
        
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i*n + j] = dx*dx + dy*dy
                dists[j*n + i] = dists[i*n + j]
        
        dists = np.sqrt(dists)
        sorted_pairs = np.argsort(dists)
        
        # Extract top two interacting circle pairs (excluding overlaps)
        pair0 = sorted_pairs[0]
        pair1 = sorted_pairs[1]
        
        # Extract circle indices for the first pair
        c0, c1 = divmod(pair0, n)
        # Extract circle indices for the second pair
        c2, c3 = divmod(pair1, n)
        
        # Create a modified constraint matrix: fix the positions of these two circles
        # This creates a topological shift, allowing the configuration to reorganize
        # For these two circles, we will allow their positions to be perturbed with 
        # a "constraint reordering force" to force a radical reconfiguration
        
        # We'll use a spatially aware perturbation with an adaptive factor based on their radii
        perturbation_scale = 1.5 * (radii[c0] + radii[c1]) * 0.1
        
        # Prepare for the "reconfiguration" step by isolating these circles
        # Create an initial perturbed configuration
        new_v = v.copy()
        
        # Apply spatial hashing perturbations to the top 4 interactive circles
        # Use adaptive scaling based on their radii and interaction distance
        hash_map = np.random.rand(n, 2) * 0.05
        for idx in [c0, c1, c2, c3]:
            new_v[3*idx] += hash_map[idx, 0] * (radii[idx] / np.mean(radii))
            new_v[3*idx+1] += hash_map[idx, 1] * (radii[idx] / np.mean(radii))
        
        # Apply a "recomposition" constraint on the first two interacting circles to force 
        # spatial reordering, which may lead to an improved configuration
        recompose_v = new_v.copy()
        
        # This force applies to the first two interacting circles, allowing them to "reorder"
        # This creates a topological instability to escape local optima
        recompose_v[3*c0] += 0.25 * perturbation_scale
        recompose_v[3*c0+1] += 0.25 * perturbation_scale
        recompose_v[3*c1] -= 0.25 * perturbation_scale
        recompose_v[3*c1+1] -= 0.25 * perturbation_scale
        
        # Re-optimized: this creates a radical topological change
        res = minimize(neg_sum_radii, recompose_v, 
                       method="SLSQP", bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-3})
        
        # Now, identify the circle with the least geometric constraint (least "locked" in position)
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            
            # Calculate all pairwise distances to identify the least constrained circle
            pairwise_dists = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i < j:
                        dx = centers[i, 0] - centers[j, 0]
                        dy = centers[i, 1] - centers[j, 1]
                        pairwise_dists[i, j] = np.sqrt(dx*dx + dy*dy)
                    else:
                        pairwise_dists[i, j] = np.inf
            
            # Determine how much "space" each circle has, defined as the minimum distance to any other
            min_space = np.min(pairwise_dists, axis=1)
            
            # The circle with the most available space is the most expandable
            least_constrained_idx = np.argmax(min_space)
            
            # Expand all radii while preserving the "least constrained circle" as an anchor
            current_sum = np.sum(radii)
            growth_factor = 0.007  # Targeted expansion of total sum
            
            # Compute a vector of expansion amounts, favoring least constrained circle
            # Apply a radial growth that is more aggressive for the least constrained circle
            new_radii = radii.copy()
            expansion = (current_sum + growth_factor) - current_sum
            
            # Distribute expansion more generously to least constrained circle
            for i in range(n):
                if i != least_constrained_idx:
                    # Apply controlled expansion, using spatial awareness
                    neighbor_min_dist = np.min(pairwise_dists[i, :], where=(pairwise_dists[i, :] != np.inf))
                    # Scale factor based on proximity to other circles
                    scale = np.clip((neighbor_min_dist - radii[i]) / (radii[i] + 1e-15), 0, 1) ** 0.5
                    expansion_i = expansion * 1.2 * scale
                    new_radii[i] += expansion_i
                else:
                    # Anchor the least constrained circle to prevent drift
                    # Apply a slight additional expansion based on available radius
                    available_growth = np.clip(1.0 / radii[i] - 1, 0, 1)
                    new_radii[i] += expansion * 1.5 * available_growth
            
            # Apply a validation check before proceeding
            # Check if new radii would cause overlaps with any other circle
            valid = True
            new_centers = np.column_stack([v[0::3], v[1::3]])
            new_radii_array = new_radii
            
            for i in range(n):
                for j in range(i + 1, n):
                    dx = new_centers[i, 0] - new_centers[j, 0]
                    dy = new_centers[i, 1] - new_centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < (new_radii_array[i] + new_radii_array[j]) - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Apply the expanded radii to the decision vector
                v_new = v.copy()
                v_new[2::3] = new_radii
                # Re-optimize under the same constraints with the updated radii
                res = minimize(neg_sum_radii, v_new, 
                               method="SLSQP", bounds=bounds,
                               constraints=cons,
                               options={"maxiter": 400, "ftol": 1e-10, "eps": 1e-3})
            else:
                # If validation failed, scale back the expansion
                # Apply a "soft" expansion that maintains constraint validity
                # Use a fractional approach to maintain validity
                scaling_factor = 0.9
                for i in range(n):
                    new_radii[i] = np.clip(new_radii[i] * scaling_factor, 1e-4, 0.5)
                # Apply the adjusted radii
                v_new = v.copy()
                v_new[2::3] = new_radii
                res = minimize(neg_sum_radii, v_new, 
                               method="SLSQP", bounds=bounds,
                               constraints=cons,
                               options={"maxiter": 400, "ftol": 1e-10, "eps": 1e-3})
                
    # Final refinement phase with adaptive scaling and tight tolerances
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Apply final validation, especially ensuring no circle exceeds square
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if (x - r < -1e-12 or x + r > 1 + 1e-12
                    or y - r < -1e-12 or y + r > 1 + 1e-12):
                # If any circle is outside, reposition it within bounds
                centers[i, 0] = max(min(x, 1.0), 0.0)
                centers[i, 1] = max(min(y, 1.0), 0.0)
                radii[i] = np.clip(r, 1e-4, 0.5)
                v[3*i] = centers[i, 0]
                v[3*i+1] = centers[i, 1]
                v[3*i+2] = radii[i]
        
        # Another optimization pass for tight tolerances
        res = minimize(neg_sum_radii, v, 
                       method="SLSQP", bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-6})
    
    # Final extraction of result
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())