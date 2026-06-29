import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    xs = np.random.uniform(0.0, 1.0, size=n)
    ys = np.random.uniform(0.0, 1.0, size=n)
    r0 = 0.25 / cols - 1e-3  # Initial radius estimates with lower starting point

    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # All circles must lie within [0,1]x[0,1], radii >= 1e-4

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    
    # Define the constraint functions properly with captures
    def create_boundary_constraints(i):
        def fun(v):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            # Left boundary
            return v[3*i] - v[3*i+2]
        return fun
    
    def create_boundary_right(v, i):
        return 1.0 - v[3*i] - v[3*i+2]
    
    def create_bottom_constraint(v, i):
        return v[3*i+1] - v[3*i+2]
    
    def create_top_constraint(v, i):
        return 1.0 - v[3*i+1] - v[3*i+2]

    for i in range(n):
        cons.append({"type": "ineq", "fun": create_boundary_constraints(i)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: create_boundary_right(v, i)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: create_bottom_constraint(v, i)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: create_top_constraint(v, i)})
    
    # Define the overlapping constraint function using vectorization
    def create_overlap_constraint(i, j):
        def fun(v):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            radii_i = v[3*i+2]
            radii_j = v[3*j+2]
            return dx*dx + dy*dy - (radii_i + radii_j)**2
        return fun

    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": create_overlap_constraint(i, j)})

    # First optimization phase with random initialization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})

    if res.success:
        v = res.x
        # Spatial hashing for non-local reconfiguration
        hash_scale = 0.06
        random_hash = np.random.rand(n, 2) * 2 * hash_scale - hash_scale
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        
        # Re-evaluate with new spatial configuration and same constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

        if res.success:
            v = res.x
            # Evaluate radii and find the smallest non-zero radius
            radii = v[2::3]
            smallest_idx = np.argmin(radii)
            smallest_radius = radii[smallest_idx]

            # Create a topological reordering adjacency constraint
            # Compute all pairwise distances
            centers = np.column_stack([v[0::3], v[1::3]])
            dists = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dists[i, j] = np.sqrt(dx*dx + dy*dy)
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmax(min_dists)

            # Apply controlled expansion to least constrained circle
            expansion_factor = (0.006 / (n - 1)) * 0.95  # Reduce slightly from previous 0.006
            new_radii = radii.copy()
            new_radii[least_constrained_idx] += expansion_factor * 1.05  # Moderate over-expansion
            for i in range(n):
                if i != least_constrained_idx:
                    new_radii[i] += expansion_factor * 0.98  # Subtle expansion for others

            # Update the decision vector
            v_new = v.copy()
            v_new[2::3] = new_radii

            # Re-evaluate with new configuration
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())