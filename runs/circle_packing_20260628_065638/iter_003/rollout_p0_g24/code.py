import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

def run_packing():
    n = 26
    # Generate initial points using Voronoi tessellation for better distribution
    np.random.seed(42)
    initial_points = np.random.rand(n, 2)
    vor = Voronoi(initial_points)
    # Extract vertices of the Voronoi diagram and use them as initial points
    # We take the vertices as the initial points, but ensure we have exactly 26
    # For simplicity, we use the initial random points instead of Voronoi vertices
    # This is a placeholder for a more sophisticated Voronoi-based initialization
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    r0 = 0.05  # Start with a small radius

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

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    
    # Local refinement step: perturb the most isolated circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Compute isolation based on distance to other circles
        isolation = np.zeros(n)
        for i in range(n):
            dist = np.min(np.sqrt(np.sum((centers[i] - centers[j])**2) for j in range(n) if j != i))
            isolation[i] = dist
        max_isolation_index = np.argmax(isolation)
        # Adjust radius and position of the most isolated circle
        v[3*max_isolation_index + 2] += 0.001
        v[3*max_isolation_index + 0] += 0.005
        v[3*max_isolation_index + 1] += 0.005
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-9})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())