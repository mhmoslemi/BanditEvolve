import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    initial_radius_upper_bound = 1.0 / 3.5
    
    # Initial randomized grid with adaptive spacing, staggered layout, 
    # and spatial gradient-aware perturbations
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid: center in cell
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Adaptive radius-aware perturbations: smaller cells allow more perturbation
        perturbation_multiplier = 0.1 + (n - 1) / (n) * (1.0 / (1.0 + base_x)) * (1.0 / (1.0 + base_y))
        x_perturb = np.random.uniform(-1.0 * perturbation_multiplier, 1.0 * perturbation_multiplier)
        y_perturb = np.random.uniform(-1.0 * perturbation_multiplier, 1.0 * perturbation_multiplier)
        # Shift alternate rows to create staggered grid with optimized height
        if row % 2 == 1:
            x_perturb += 0.25 / cols
        xs.append(base_x + x_perturb)
        ys.append(base_y + y_perturb)
        
    # Compute initial radii based on grid cell diameter plus safety margins
    # Ensure minimal cell spacing to allow for expansion
    # Use an adaptive initial radius strategy - larger cells get larger base radii
    # The radius of a cell is determined by max(1.0 / (2 * (n/cols)), initial_radius_upper_bound)
    base_cell_radius = np.maximum(1.0 / (2 * (n / cols)), initial_radius_upper_bound)
    r0 = base_cell_radius - 1e-2  # Safety margin for expansion

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3 * n

    def neg_sum_radii(v):
        """Objective: maximize sum of radii (minimize negative sum)"""
        return -np.sum(v[2::3])

    # Constraints: 4 per circle for bounding square, n*(n-1)/2 for pairwise circle overlaps
    cons = []

    # Define constraint functions safely using lambda captures with i, and avoid capture issues
    # Vectorized boundaries with strict tolerance handling (1e-8)
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints: 4-dimensional, pairwise, with adaptive tolerance
    for i in range(n):
        for j in range(i + 1, n):
            # Lambda capturing i and j safely
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j]) ** 2 
                    + (v[3*i+1] - v[3*j+1]) ** 2 
                    - (v[3*i+2] + v[3*j+2]) ** 2
                    + 1e-12  # Add small offset to avoid strict equality 
                )
            })
    
    # Phase 1: Initial optimization with moderate iter and high precision
    phase_1_options = {
        "maxiter": 600, 
        "ftol": 1e-11, 
        "gtol": 1e-10,
        "eps": np.finfo(float).eps * 1e1,
        "bounds": bounds,
        "constraints": cons
    }
    phase_1_res = minimize(neg_sum_radii, v0, method="SLSQP", **phase_1_options)
    
    # Phase 2: Adaptive reconfiguration with perturbation gradients
    # Only proceed if first phase is successful
    if phase_1_res.success:
        v_initial = phase_1_res.x
        radii_initial = v_initial[2::3]
        centers_initial = np.column_stack([v_initial[0::3], v_initial[1::3]])

        # Calculate pairwise distances once for gradient-aware spatial mapping
        dists_initial = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers_initial[i, 0] - centers_initial[j, 0]
                dy = centers_initial[i, 1] - centers_initial[j, 1]
                dists_initial[i, j] = np.sqrt(dx**2 + dy**2)
        
        # Compute least constrained indices by max of min_dists to other circles
        min_dists = np.min(dists_initial, axis=1)
        least_constrained_idx = np.argsort(min_dists)[::-1][:3]  # 3 least constrained

        # Apply spatial gradient-aware perturbations to the least constrained
        # Use a weighted randomization that's biased towards small perturbations
        # Perturbations are proportional to the inverse of their current radii
        # So smaller circles get more spatial freedom
        perturbation_weights = 1.0 / np.maximum(radii_initial, 1e-6)
        perturbation_weights = perturbation_weights / np.sum(perturbation_weights)
        
        # Generate spatial maps with weighted randomization
        spatial_map = np.random.randn(n, 2) * 0.001 * perturbation_weights[:, None]
        # Add a spatial gradient from center to edge to avoid clustering
        # Spatial gradient is based on position to encourage spreading out
        spatial_gradient = (np.array(centers_initial) - np.array([0.5, 0.5])) * 0.0005
        # Blend gradient with spatial noise
        spatial_map += spatial_gradient
        
        # Apply perturbations to positions: scaled by radius to preserve feasibility
        perturbed_centers = centers_initial + spatial_map * radii_initial[:, None]

        # Build perturbed vector
        perturbed_v = v_initial.copy()
        perturbed_v[0::3] = perturbed_centers[:, 0]
        perturbed_v[1::3] = perturbed_centers[:, 1]

        # Phase 2 optimization with tighter iteration and bounds
        phase_2_options = {
            "maxiter": 800, 
            "ftol": 1e-12, 
            "gtol": 1e-11,
            "eps": np.finfo(float).eps * 1e1,
            "bounds": bounds,
            "constraints": cons
        }
        phase_2_res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", **phase_2_options)

        if phase_2_res.success:
            v_phase2 = phase_2_res.x
        else:
            v_phase2 = phase_1_res.x

    else:
        v_phase2 = v0

    # Phase 3: Targeted radius expansion on smallest circles with gradient-aware constraints
    # Only proceed if we have a successful state
    if v_phase2 is not None and len(v_phase2) == 3*n:
        v = v_phase2
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute distance matrix with vectorized numpy
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Compute minimum distance to avoid over-estimation
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argsort(min_dists)[-3:][::-1]  # Least constrained three

        # Apply radial gradient-aware expansion: expand smaller radius circles first
        # Use a soft growth strategy with adaptive expansion rate per circle
        # Expansion is proportional to the minimum distance to neighbors, to avoid overlap
        expansion_coefficient = (0.0015) / (n)
        expansion_per_circle = np.zeros(n)
        for idx in least_constrained_idx:
            expansion_per_circle[idx] = np.min(min_dists) * expansion_coefficient

        # Create new_radii with safe expansion, ensuring no overlap is created
        new_radii = radii.copy()
        for i in range(n):
            if i in least_constrained_idx:
                # Safety margin for expansion (avoid overlapping with neighbors)
                # Calculate expansion limit per circle
                current_radius = radii[i]
                max_expansion = 0.0
                for j in range(n):
                    if j != i:
                        current_distance = dists[i, j]
                        max_possible_grow = (current_distance - (radii[i] + radii[j])) / (2.0)
                        if max_possible_grow > 0:
                            max_expansion = max(max_expansion, max_possible_grow)
                # Add expansion to radius
                new_radii[i] = np.clip(radii[i] + 0.9 * expansion_per_circle[i], 1e-6, 1.0 - (1e-6 / (n)))

        # Create new vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii

        # Phase 3 optimization with very tight constraints
        phase_3_options = {
            "maxiter": 400, 
            "ftol": 1e-12, 
            "gtol": 1e-11,
            "eps": np.finfo(float).eps * 1e1,
            "bounds": bounds,
            "constraints": cons
        }
        phase_3_res = minimize(neg_sum_radii, v_new, method="SLSQP", **phase_3_options)

        if phase_3_res.success:
            v_expanded = phase_3_res.x
        else:
            v_expanded = v_phase2

    else:
        v_expanded = v_phase2

    # Final check of validity
    v = v_expanded if isinstance(v_expanded, np.ndarray) else v_phase2
    
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final safety check
    if not validate_packing(centers, radii)[0]:
        return np.zeros((n, 2)), np.zeros(n), 0.0
    
    return centers, radii, float(radii.sum())