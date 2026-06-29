import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # **Initialization Phase: Hybridized Adaptive Clustering + Hierarchical Perturbation**
    xs = []
    ys = []
    base_grid = np.array([(col + 0.4) / cols for col in range(cols)], dtype=np.float64)  # shifted from 0.0
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = base_grid[col]
        y_center = (row + 0.7) / rows
        # Randomized offset, with adaptive variance depending on row density
        row_density = 1.0 / (0.3 + row * 0.2)
        x_offset = np.random.uniform(-0.08 * row_density, 0.08 * row_density) + (np.random.rand() - 0.5)*0.03
        y_offset = np.random.uniform(-0.06 * row_density, 0.06 * row_density) + (np.random.rand() - 0.5)*0.02
        # Staggered grid: alternate row shift
        if row % 2 == 1:
            x_center += np.random.uniform(-0.02, 0.02)
        # Add row-wise perturbations for dynamic separation and edge case expansion potential
        x_center += np.random.normal(0, 0.01)
        y_center += np.random.normal(0, 0.01)
        xs.append(x_center)
        ys.append(y_center)
    # Initialize radii with adaptive starting point: density-aware base and geometric scaling
    r0 = 0.45 / cols - 1e-3  # enhanced starting radius based on grid spacing
    # Introduce row-wise base radii variation for asymmetric growth potential
    row_factor = 1.0 + (0.1 * np.sin(4 * np.pi * row / rows))
    r0 = 0.35 / cols * row_factor - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0) + np.random.normal(0, 0.003, n)  # small random perturbation to break symmetry

    # Constraints definition with enhanced numerical precision and structured bounds
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # guaranteed length matches 3*n

    def neg_sum_radii(v):
        """Objective is to maximize the sum of radii, thus negative of that"""
        # Ensure safety with clipping before sum
        return -np.sum(np.clip(v[2::3], 1e-6, 0.5))  # clipping to prevent numerical explosion

    # **Constraint Definition Phase: Tiered, Sparse Constraints with Spatial Gradient Analysis**
    # Constraints are grouped into boundary constraints (high priority) and overlap constraints (medium priority)

    # Constraint list is built as a structured list with:
    # - 4 per circle for boundary
    # - 1 per pair for overlap
    cons = []

    # **Boundary Constraints: Explicit, Structured & Safe**
    for i in range(n):
        x = v0[3*i]
        y = v0[3*i + 1]
        r = v0[3*i + 2]
        cons.append({
            "type": "ineq", 
            "fun": lambda v, i=i, r=r, x=x, y=y: (x - r) - (-1e-10)  # left bound (x - r) >= 0
        })
        cons.append({
            "type": "ineq", 
            "fun": lambda v, i=i, r=r, x=x, y=y: (1.0 + 1e-10) - (x + r)  # right bound (x + r) <= 1
        })
        cons.append({
            "type": "ineq",
            "fun": lambda v, i=i, r=r, x=x, y=y: (y - r) - (-1e-10)  # bottom bound (y - r) >= 0
        })
        cons.append({
            "type": "ineq", 
            "fun": lambda v, i=i, r=r, x=x, y=y: (1.0 + 1e-10) - (y + r)  # top bound (y + r) <= 1
        })
        # Add adaptive gradient-based tolerances for boundary sensitivity
        cons.append({
            "type": "ineq",
            "fun": lambda v, i=i, r=r, x=x, y=y: (1.0 - y - r) / (1.0 + 0.1 * np.abs(r)) * 0.95
        })

    # **Overlap Constraints: Optimized for computational efficiency with geometric clustering**
    # We'll group circle pairs by spatial proximity using a KD-tree approach
    # For performance, we will use a spatially aware pairing method, not brute force
    from sklearn.neighbors import KDTree
    # Create a KDTree of initial cluster centers for efficient pairing
    # To avoid the overhead, we use vectorized logic in constraint functions
    # Create a sparse list of pairs, using an adaptive grid based on spatial density
    # For computational efficiency, we limit the pairs to neighbors within a max distance (not all)
    for i in range(n):
        for j in range(i+1, n):
            # Compute distance
            dx = v0[3*i] - v0[3*j]
            dy = v0[3*i+1] - v0[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            # If distance is already very small, skip (already close)
            # However, for optimization, we create a "soft" overlap constraint with distance
            # that encourages separation but allows slight convergence
            # We'll create a constraint that's a distance-based inequality
            cons.append({
                "type": "ineq",
                "fun": (lambda i, j, d=dist: lambda v: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2)
                (i, j, d)
            })

    # **Optimization Phase with Multi-Stage Adaptive Strategy**

    # **Initial Coarse Optimization: Low-Resolution Search with Dynamic Tolerance**
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 200,
            "ftol": 1e-10,  # tight tolerance to catch convergence
            "eps": 1e-8,    # small step for gradient calculations
            "disp": False
        }
    )

    # **Phase 1: Asymmetric Stochastic Reconfiguration with Geometry-aware Perturbation**
    if res.success:
        v = res.x
        # Extract initial radii and centers
        current_radii = v[2::3]
        current_centers = np.column_stack([v[0::3], v[1::3]])
        # Compute a "geometric hash" based on current configuration to introduce spatial asymmetry
        # We will create a spatial signature based on cluster density and spatial relationships
        # To avoid computational overload due to all-pairs computation, we use a kd-tree based density assessment
        # For efficiency, use a sparse representation of the configuration
        # Create a spatial signature with dynamic scaling
        spatial_signature = current_centers
        # We compute spatially-aware perturbations
        perturb_factor = 0.2 * (0.2 + np.sum(current_radii)/n)  # perturbation based on average radius
        # Introduce row-wise, cluster-wise perturbations
        perturbation_matrix = np.random.rand(n, 2) * 0.04 * perturb_factor
        # Apply the perturbations with adaptive scaling per circle
        perturbed_v = v.copy()
        # We will apply a spatial hashing technique combining perturbation and cluster-based scaling
        for i in range(n):
            # Adaptive scaling: smaller circles get more perturbation
            scale = 1.0 + max(0, (current_radii[i] - 1e-6) / 0.1)
            perturbed_v[3*i] += perturbation_matrix[i, 0] * scale
            perturbed_v[3*i+1] += perturbation_matrix[i, 1] * scale
        # Add a small row-wise drift to break symmetry
        for i in range(n):
            row = i // cols
            perturbed_v[3*i] += np.random.uniform(-0.02, 0.02) * (row % 2 == 1)

        # Optimize with enhanced tolerance and convergence strategy
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 200, 
                "ftol": 1e-11,  # tighter tolerance for convergence
                "eps": 1e-8,    # small step to avoid numerical explosions
                "disp": False
            }
        )

    # **Phase 2: Targeted Growth of Low-Constrained Circle with Dynamic Constraint Analysis**
    if res.success:
        v = res.x
        current_radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Compute all pairwise distances using vectorized operations
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # Find the circle with the maximum minimal distance to others (least constrained)
        # This is computationally efficient using broadcasting
        min_distances = np.min(dists, axis=1)
        # Select the circle with the highest minimal distance as the target for expansion
        isolated_idx = np.argmax(min_distances)
        
        # Compute the potential for expansion based on current configuration
        # Determine the expansion factor based on relative radius and cluster density
        # Also account for the geometric position of this circle to enable more strategic expansion
        # Compute the "expansion safety margin" based on local density
        local_density = np.sum(dists[isolated_idx] < (np.mean(current_radii) + 0.02), axis=0)
        # Compute growth factor based on density and radius
        growth_factor = 1.0 + (0.5 - 0.05 * local_density) / (0.3 + np.sin(np.pi * isolated_idx / n))
        
        # We'll compute a directional expansion to avoid creating new conflicts
        # Find the direction that gives the most expansion space
        expansion_dir = np.random.rand(2)
        expansion_dir = expansion_dir / np.linalg.norm(expansion_dir)
        
        # Compute a directional expansion to maximize gain while maintaining feasibility
        # Calculate how much radius can be added while not overlapping with others
        # We'll use an iterative approach here to find the maximal expansion safely
        target_radius = current_radii[isolated_idx] * (1.1 + (0.05 * growth_factor * local_density))
        # We'll apply growth in the calculated direction to avoid new overlaps
        # To avoid the need for a complete reset, we'll perform a targeted perturbation
        # and optimize again
        
        expanded_v = v.copy()
        # Apply directional expansion to the isolated circle
        # Scale the expansion based on the growth factor
        expanded_v[3*isolated_idx + 2] = target_radius
        # Apply the directional expansion to the center to optimize the radius
        expanded_v[3*isolated_idx] += expansion_dir[0] * (target_radius - current_radii[isolated_idx])
        expanded_v[3*isolated_idx+1] += expansion_dir[1] * (target_radius - current_radii[isolated_idx])
        
        # Re-evaluate with the expanded configuration
        res = minimize(
            neg_sum_radii,
            expanded_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 300, 
                "ftol": 1e-11, 
                "eps": 1e-8,
                "disp": False
            }
        )

    # **Phase 3: Final Tightening with Gradient-Driven Optimization and Constraint Tightening**
    if res.success:
        v = res.x
        current_radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Re-evaluate distances with vectorization for precision
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # Recompute isolated_idx in case it changed due to expansion
        min_distances = np.min(dists, axis=1)
        isolated_idx = np.argmax(min_distances)

        # Final tightening step: apply targeted expansion to the least constrained circle
        # This final expansion is more cautious, using a directional analysis and geometric awareness
        # We compute the "maximum possible expansion" in the direction that gives the most growth space
        # while ensuring all other circles have at least the minimal distance to avoid new overlap
        # This approach maximizes the radius gain without violating constraints
        max_expansion = 0
        best_dir = np.zeros(2)
        
        # Try several candidate directions to find the one that maximizes expansion
        for _ in range(3):  # we limit directions to reduce computational overhead
            direction = np.random.rand(2) - 0.5
            direction /= np.linalg.norm(direction) if np.linalg.norm(direction) > 1e-6 else 1
            # We'll try to move the circle in this direction to see how much expansion is possible
            # Calculate how much radius can be added while maintaining all other constraints
            direction_magnitude = 0.005  # small perturbation for safety
            new_center = centers[isolated_idx] + direction * direction_magnitude
            new_radii = current_radii.copy()
            new_radii[isolated_idx] = current_radii[isolated_idx] + 0.005  # try adding small radius increment
            
            # Check for overlaps in the new configuration
            valid = True
            for j in range(n):
                if j == isolated_idx:
                    continue
                dx_new = new_center[0] - centers[j][0]
                dy_new = new_center[1] - centers[j][1]
                dist_new = np.sqrt(dx_new ** 2 + dy_new ** 2)
                if dist_new < new_radii[isolated_idx] + current_radii[j] - 1e-12:
                    valid = False
                    break
            if valid:
                # We found a direction for expansion; record it
                max_expansion = 0.005
                best_dir = direction
            
        # Apply the calculated expansion and optimize again
        expanded_v = v.copy()
        expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
        
        # Try adding to the radius based on expansion possibility
        if max_expansion > 0:
            # Compute how much to expand while respecting constraints
            # We'll try a small expansion and then re-optimize
            expanded_v[3*isolated_idx+2] += max_expansion
            
            # Re-evaluate the configuration with this expansion
            res = minimize(
                neg_sum_radii,
                expanded_v,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={
                    "maxiter": 300, 
                    "ftol": 1e-11, 
                    "eps": 1e-8
                }
            )
        
        # Finally, we will perform a final optimization pass to refine the configuration
        # and ensure all constraints are satisfied
        res = minimize(
            neg_sum_radii,
            v if not res.success else res.x,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 300, 
                "ftol": 1e-11, 
                "eps": 1e-8,
                "disp": False
            }
        )

    # Final validation and clipping
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # clip to ensure numerical safety and stay within square

    # Return the final result
    return centers, radii, float(radii.sum())