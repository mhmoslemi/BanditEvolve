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

    def create_overlap_constraints():
        overlap_cons = []
        for i in range(n):
            for j in range(i + 1, n):
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                overlap_cons.append({"type": "ineq", "fun": constraint_func})
        return overlap_cons

    cons += create_overlap_constraints()

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    constraint_tightness = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                constraint_tightness[i] += (v[3*i+2] + v[3*j+2] - dist)
                constraint_tightness[j] += (v[3*i+2] + v[3*j+2] - dist)
    
    sorted_indices = np.argsort(constraint_tightness)
    permuted_v = np.zeros_like(v)
    for i, idx in enumerate(sorted_indices):
        permuted_v[3*i] = v[3*idx]
        permuted_v[3*i+1] = v[3*idx+1]
        permuted_v[3*i+2] = v[3*idx+2]
    
    res = minimize(neg_sum_radii, permuted_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else permuted_v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    log_v = np.log(v + 1e-10)
    log_v[0::3] = (log_v[0::3] - np.min(log_v[0::3])) / (np.max(log_v[0::3]) - np.min(log_v[0::3]))
    log_v[1::3] = (log_v[1::3] - np.min(log_v[1::3])) / (np.max(log_v[1::3]) - np.min(log_v[1::3]))
    log_v[2::3] = (log_v[2::3] - np.min(log_v[2::3])) / (np.max(log_v[2::3]) - np.min(log_v[2::3]))
    
    distorted_v = np.copy(log_v)
    distorted_v[0::3] *= 1.2
    distorted_v[1::3] *= 1.2
    distorted_v[2::3] *= 0.8

    res = minimize(neg_sum_radii, distorted_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else distorted_v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    def penalty(v):
        sum_penalty = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                    sum_penalty += max(0, (v[3*i+2] + v[3*j+2] - dist) ** 2)
        return sum_penalty

    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    if np.sum(radii) > 0:
        sorted_indices = np.argsort(radii)
        small_circle_indices = sorted_indices[:10]
        perturbation = 0.01
        for idx in small_circle_indices:
            i = idx
            v[3*i] += np.random.uniform(-perturbation, perturbation)
            v[3*i+1] += np.random.uniform(-perturbation, perturbation)
            v[3*i+2] = np.clip(v[3*i+2], 1e-6, 0.5)
        res = minimize(lambda v: -np.sum(v[2::3]) + 100 * penalty(v), v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 300, "ftol": 1e-9})
        v = res.x if res.success else v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)

    if validate_packing(centers, radii)[0]:
        for i in range(n):
            new_radius = radii[i] + 1e-6
            overlap = False
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < new_radius + radii[j] - 1e-12:
                    overlap = True
                    break
            if not overlap:
                v[3*i+2] = new_radius
                radii = np.clip(v[2::3], 1e-6, None)
                centers = np.column_stack([v[0::3], v[1::3]])

    if validate_packing(centers, radii)[0]:
        for i in range(n):
            new_radius = radii[i] + 1e-8
            overlap = False
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < new_radius + radii[j] - 1e-12:
                    overlap = True
                    break
            if not overlap:
                radii[i] = new_radius

    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-12, "eps": 1e-12})
    v = res.x if res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    sorted_by_radius = np.argsort(radii)
    smallest_indices = sorted_by_radius[:3]
    original_positions = v[3*smallest_indices[0]+0:3*smallest_indices[0]+3].copy()
    original_positions = np.vstack((original_positions[0::3], original_positions[1::3])).copy()
    
    temp_x = v[3*smallest_indices[0]+0]
    temp_y = v[3*smallest_indices[0]+1]
    v[3*smallest_indices[0]+0] = v[3*smallest_indices[1]+0]
    v[3*smallest_indices[0]+1] = v[3*smallest_indices[1]+1]
    v[3*smallest_indices[1]+0] = temp_x
    v[3*smallest_indices[1]+1] = temp_y
    temp_x = v[3*smallest_indices[0]+0]
    temp_y = v[3*smallest_indices[0]+1]
    v[3*smallest_indices[0]+0] = v[3*smallest_indices[2]+0]
    v[3*smallest_indices[0]+1] = v[3*smallest_indices[2]+1]
    v[3*smallest_indices[2]+0] = temp_x
    v[3*smallest_indices[2]+1] = temp_y

    def modified_neg_sum_radii(v):
        return -np.sum(v[2::3]) + 100 * np.sum(v[2::3] ** 2)
    
    res = minimize(modified_neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    if validate_packing(centers, radii)[0]:
        for i in range(n):
            new_radius = radii[i] + 1e-6
            overlap = False
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < new_radius + radii[j] - 1e-12:
                    overlap = True
                    break
            if not overlap:
                v[3*i+2] = new_radius
                radii = np.clip(v[2::3], 1e-6, None)
                centers = np.column_stack([v[0::3], v[1::3]])

    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-12, "eps": 1e-12})
    v = res.x if res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Apply geometric phase shift mutation
    phase_shift_v = np.copy(v)
    phase_shift_v[0::3] *= np.exp(np.random.uniform(-0.1, 0.1, n))
    phase_shift_v[1::3] *= np.exp(np.random.uniform(-0.1, 0.1, n))
    phase_shift_v[2::3] *= np.exp(np.random.uniform(-0.1, 0.1, n))

    res = minimize(neg_sum_radii, phase_shift_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    phase_shift_v = res.x if res.success else phase_shift_v
    phase_shift_centers = np.column_stack([phase_shift_v[0::3], phase_shift_v[1::3]])
    phase_shift_radii = np.clip(phase_shift_v[2::3], 1e-6, None)

    if validate_packing(phase_shift_centers, phase_shift_radii)[0] and np.sum(phase_shift_radii) > np.sum(radii):
        v = phase_shift_v
        centers = phase_shift_centers
        radii = phase_shift_radii

    return centers, radii, float(radii.sum())