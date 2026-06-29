import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with dynamic spacing and randomized offset
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        col_width = 1.0 / cols
        row_height = 1.0 / rows
        x_center = col * col_width + col_width / 2 + np.random.uniform(-0.03, 0.03)
        y_center = row * row_height + row_height / 2 + np.random.uniform(-0.03, 0.03)
        if row % 2 == 1:
            x_center += col_width / 2
        xs.append(x_center)
        ys.append(y_center)
    
    r0 = 0.32 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints using numpy broadcasting
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j])**2 + 
                    (v[3*i+1] - v[3*j+1])**2 - 
                    (v[3*i+2] + v[3*j+2])**2
                )
            })

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-10})
    
    # Primary optimization phase
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances with numpy broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1, :]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Shake phase: perturbing constrained circles
        for shake_step in range(3):
            # Identify least constrained circles
            min_distances = np.min(dists, axis=1)
            least_constrained_idx = np.argsort(min_distances)[::-1][:5]  # 5 most unconstrained
            radii[least_constrained_idx] *= 1.005  # Slight radius expansion
            
            # Create a new decision vector with increased radii
            new_v = v.copy()
            new_v[2::3] = radii
            
            # Reoptimize with increased radii for constrained circles
            res = minimize(
                neg_sum_radii,
                new_v,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 300, "ftol": 1e-11}
            )
            
            if not res.success:
                break
        
        # Final optimization after shaking
        v = res.x if res.success else v0
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Final radius expansion without overlap
        # Compute pairwise distances with numpy broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1, :]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with maximum expansion potential
        min_dists = np.min(dists, axis=1)
        expansion_idx = np.argmin(min_dists)  # smallest minimum distance = most constrained
        
        # Calculate maximum possible expansion for this circle
        max_expansion = (np.min(dists[expansion_idx]) - 2*radii[expansion_idx]) / 2
        if max_expansion > 0:
            # Apply expansion and reoptimize
            new_radii = radii.copy()
            new_radii[expansion_idx] += max_expansion
            new_v = v.copy()
            new_v[2::3] = new_radii
        
            res = minimize(
                neg_sum_radii,
                new_v,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 300, "ftol": 1e-11}
            )
        
        v = res.x if res.success else v0
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())