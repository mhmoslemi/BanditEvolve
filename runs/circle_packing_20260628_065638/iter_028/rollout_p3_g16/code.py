import numpy as np
from functools import partial

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # Initialize with a more balanced staggered grid and randomized spatial perturbation
    xs = []
    ys = []
    base_r = 0.35 / cols
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add spatial randomness with more aggressive perturbations for better distribution
        noise = np.random.uniform(-0.06, 0.06, 2)
        x = x_center + noise[0]
        y = y_center + noise[1]
        # Staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radius based on spacing
    r0 = base_r - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using closure binding
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": partial(lambda v, idx: v[3*idx] - v[3*idx+2], idx=i)})
        cons.append({"type": "ineq", "fun": partial(lambda v, idx: 1.0 - v[3*idx] - v[3*idx+2], idx=i)})
        cons.append({"type": "ineq", "fun": partial(lambda v, idx: v[3*idx+1] - v[3*idx+2], idx=i)})
        cons.append({"type": "ineq", "fun": partial(lambda v, idx: 1.0 - v[3*idx+1] - v[3*idx+2], idx=i)})
    
    # Vectorized overlap constraints with closure binding
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": partial(
                lambda v, i, j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                - (v[3*i+2] + v[3*j+2])**2, 
                i=i, j=j
            )})

    # Initial optimization with high convergence parameters
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1500, "ftol": 1e-10, "gtol": 1e-10}
    )
    
    # Asymmetric reconfiguration with spatial constraint-aware perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Calculate spatial density map for perturbation
        dists = np.zeros((n, n))
        x_coords = centers[:, 0]
        y_coords = centers[:, 1]
        dx = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
        dy = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
        dists = np.sqrt(dx**2 + dy**2)
        dists = np.abs(dists - (radii[:, np.newaxis] + radii[np.newaxis, :]))
        
        # Calculate least constrained circles for expansion
        min_distances = np.min(dists, axis=1)
        least_constrained_idx = np.argsort(min_distances)[::-1]  # Sort descending

        # Create a spatial hash map scaled by distance metrics
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbation_factor = min_distances / np.max(min_distances)
        perturbed_v = v.copy()
        for i in range(n):
            # Scale perturbation with spatial density and radius
            perturb_scale = (perturbation_factor[i] + 1) * (radii[i] / np.mean(radii))
            perturbed_v[3*i] += spatial_hash[i, 0] * perturb_scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * perturb_scale
        
        # Re-evaluate with new spatial configuration
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11}
        )
    
    # Targeted radius expansion based on constraint violation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute current constraint violations
        dists = np.zeros((n, n))
        x_coords = centers[:, 0]
        y_coords = centers[:, 1]
        dx = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
        dy = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
        dists = np.sqrt(dx**2 + dy**2)
        constraint_violations = dists - (radii[:, np.newaxis] + radii[np.newaxis, :])
        
        # Find circles with largest margin of safety for expansion
        margin_of_safety = np.max(np.maximum(constraint_violations, 0), axis=1)
        expansion_candidates = np.argsort(margin_of_safety)[::-1]  # Most margin first
        expansion_radius = np.mean(radii)

        # Initial expansion based on spatial expansion potential
        expansion_factor = 0.006 / (np.sum(radii) - 0.001)
        expansion_ratio = np.random.rand(n) * 1.2 + 0.8
        
        # Create a new radii vector with controlled expansion
        new_radii = radii.copy()
        for i in expansion_candidates:
            if new_radii[i] < 0.45:
                new_radii[i] += expansion_factor * expansion_ratio[i]
        
        # Create perturbation to break local minima
        perturbation = np.random.rand(n, 2) * 0.02
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation[i, 0]
            perturbed_v[3*i+1] += perturbation[i, 1]
        
        # Re-evaluate with new radii and perturbation
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11}
        )
    
    # Final optimization with enhanced convergence
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Final refinement with small additional expansion
        expansion_factor = 0.004 / (np.sum(radii) + 0.001)
        expansion_ratio = np.random.rand(n) * 1.1 + 0.9
        
        new_radii = radii.copy()
        for i in range(n):
            if new_radii[i] < 0.45:
                new_radii[i] += expansion_factor * expansion_ratio[i]
        
        # Apply expansion with constraint validation
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Re-evaluate
        res = minimize(
            neg_sum_radii,
            expanded_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11}
        )
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())