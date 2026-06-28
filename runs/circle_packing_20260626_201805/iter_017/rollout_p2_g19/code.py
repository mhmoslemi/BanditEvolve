import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

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

    def vectorized_overlap_constraint(v):
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        dx = x_centers[:, np.newaxis] - x_centers[np.newaxis, :]
        dy = y_centers[:, np.newaxis] - y_centers[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r_radii[:, np.newaxis] + r_radii[np.newaxis, :])**2
        return dist_sq - min_dist_sq

    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            overlap_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: vectorized_overlap_constraint(v)[i, j]})

    cons.extend(overlap_cons)

    # Enforce radical topological reconfiguration
    def reconfigure_layout(v):
        # Randomly assign new spatial constraints
        new_constraints = []
        for i in range(n):
            # Randomly choose constraint type: radial, angular, or bounding box
            constraint_type = np.random.choice(['radial', 'angular', 'bounding'])
            if constraint_type == 'radial':
                # Radial constraint: distance from a random center
                center_idx = np.random.choice(n)
                dist = np.random.uniform(0.1, 0.3)
                new_constraints.append((i, center_idx, dist))
            elif constraint_type == 'angular':
                # Angular constraint: angle relative to a random center
                center_idx = np.random.choice(n)
                angle = np.random.uniform(0, 2 * np.pi)
                new_constraints.append((i, center_idx, angle))
            else:
                # Bounding box constraint: position within a random sub-box
                box_size = np.random.uniform(0.1, 0.3)
                box_x = np.random.uniform(0, 1 - box_size)
                box_y = np.random.uniform(0, 1 - box_size)
                new_constraints.append((i, box_x, box_y, box_size))
        return new_constraints

    def apply_constraints(v, constraints):
        new_v = np.copy(v)
        for i, *params in constraints:
            if len(params) == 3:  # radial constraint
                center_idx, dist = params
                x_center = new_v[3*center_idx]
                y_center = new_v[3*center_idx + 1]
                r = new_v[3*i + 2]
                # Adjust position to satisfy radial constraint
                dx = x_center - new_v[3*i]
                dy = y_center - new_v[3*i + 1]
                dist_current = np.sqrt(dx**2 + dy**2)
                if dist_current < dist - 1e-6:
                    # Move closer to center
                    new_v[3*i] = x_center + (x_center - new_v[3*i]) * 0.5
                    new_v[3*i + 1] = y_center + (y_center - new_v[3*i + 1]) * 0.5
                    new_v[3*i + 2] = np.clip(r * 1.1, 1e-4, 0.5)
                elif dist_current > dist + 1e-6:
                    # Move further away
                    new_v[3*i] = x_center + (x_center - new_v[3*i]) * 0.5
                    new_v[3*i + 1] = y_center + (y_center - new_v[3*i + 1]) * 0.5
                    new_v[3*i + 2] = np.clip(r * 1.1, 1e-4, 0.5)
            elif len(params) == 4:  # bounding box constraint
                box_x, box_y, box_size = params
                # Adjust position to fit within the box
                x = new_v[3*i]
                y = new_v[3*i + 1]
                r = new_v[3*i + 2]
                if x - r < box_x:
                    new_v[3*i] = box_x + r
                elif x + r > box_x + box_size:
                    new_v[3*i] = box_x + box_size - r
                if y - r < box_y:
                    new_v[3*i + 1] = box_y + r
                elif y + r > box_y + box_size:
                    new_v[3*i + 1] = box_y + box_size - r
                new_v[3*i + 2] = np.clip(r * 1.1, 1e-4, 0.5)
        return new_v

    # Perform topological reconfiguration
    constraints = reconfigure_layout(v0)
    v = apply_constraints(v0, constraints)

    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    def local_refinement(centers, radii):
        for _ in range(100):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < radii[i] + radii[j] - 1e-6:
                        overlap = radii[i] + radii[j] - dist
                        dx /= dist
                        dy /= dist
                        centers[i] += dx * overlap * 0.5
                        centers[j] -= dx * overlap * 0.5
                        centers[i] += dy * overlap * 0.5
                        centers[j] -= dy * overlap * 0.5
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if x - r < 0:
                    centers[i, 0] = r
                elif x + r > 1:
                    centers[i, 0] = 1 - r
                if y - r < 0:
                    centers[i, 1] = r
                elif y + r > 1:
                    centers[i, 1] = 1 - r
        return centers, radii

    centers, radii = local_refinement(centers, radii)

    return centers, radii, float(radii.sum())