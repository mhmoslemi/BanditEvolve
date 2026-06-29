import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    # Create initial cluster centers with randomized spatial distribution
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    # Ensure clusters are initially spaced apart
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 0.1:
                # Move one of the points to ensure spacing
                if np.random.rand() < 0.5:
                    xs[i] += 0.15
                    ys[i] += 0.15
                else:
                    xs[j] += 0.15
                    ys[j] += 0.15
    
    # Normalize positions to unit square
    xs = (xs - np.min(xs)) / (np.max(xs) - np.min(xs))
    ys = (ys - np.min(ys)) / (np.max(ys) - np.min(ys))
    
    r0 = 0.25 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Identify the most tightly packed cluster and perturb it
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Compute cluster density using Voronoi regions
        from scipy.spatial import Voronoi
        vor = Voronoi(np.column_stack([centers[0], centers[1]]))
        region_areas = [vor.point_region[i] for i in range(n)]
        max_density_index = np.argmax(region_areas)
        # Perturb the cluster to allow expansion
        v[3*max_density_index + 0] += 0.01
        v[3*max_density_index + 1] += 0.01
        v[3*max_density_index + 2] += 0.005
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Add targeted refinement for small circles and boundary conditions
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify small circles and boundary circles
        small_indices = np.argsort(radii)[:4]
        boundary_indices = []
        for i in range(n):
            x = v[3*i]
            y = v[3*i+1]
            r = v[3*i+2]
            if x < r or x > 1 - r or y < r or y > 1 - r:
                boundary_indices.append(i)
        # Combine and deduplicate indices
        perturb_indices = np.unique(np.concatenate((small_indices, boundary_indices)))
        # Apply small random perturbation to their positions
        perturbation = 0.05 * np.random.rand(len(perturb_indices) * 3)
        perturbed_v = v.copy()
        idx = 0
        for i in perturb_indices:
            perturbed_v[3*i] += perturbation[idx]
            perturbed_v[3*i+1] += perturbation[idx+1]
            perturbed_v[3*i+2] += perturbation[idx+2]
            idx += 3
        # Clip radii to ensure they stay within bounds
        perturbed_v[2::3] = np.clip(perturbed_v[2::3], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())