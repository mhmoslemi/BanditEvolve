import numpy as np

def run_packing():
    n = 26
    
    # Two-stage optimization strategy: global search followed by local optimization
    
    # Stage 1: Global search with spiral initialization for better spacing
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.linspace(0.1, 0.4, n)
    spiral_x = np.zeros(n)
    spiral_y = np.zeros(n)
    
    for i in range(n):
        r = radii[i]
        angle = angles[i]
        spiral_x[i] = 0.5 + r * np.cos(angle)
        spiral_y[i] = 0.5 + r * np.sin(angle)
    
    v0_global = np.empty(3 * n)
    v0_global[0::3] = spiral_x
    v0_global[1::3] = spiral_y
    v0_global[2::3] = np.full(n, 0.05)  # Initial guess for radii

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons_global = []
    for i in range(n):
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons_global.append({"type": "ineq", "fun": constraint_func})

    # First optimization stage: global search
    res_global = minimize(neg_sum_radii, v0_global, method="SLSQP", bounds=bounds,
                         constraints=cons_global, options={"maxiter": 500, "ftol": 1e-9})
    
    v_global = res_global.x if res_global.success else v0_global
    
    # Stage 2: Local optimization using L-BFGS-B for fine-tuning
    v_local = v_global.copy()
    v_local[2::3] = np.clip(v_local[2::3], 1e-6, 0.5)  # Ensure radii are within bounds
    
    res_local = minimize(neg_sum_radii, v_local, method="L-BFGS-B", bounds=bounds,
                         constraints=cons_global, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res_local.x if res_local.success else v_global
    
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())