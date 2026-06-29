import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.375 / cols - 1e-2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

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

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Induce major geometric shift with adaptive spatial hashing reconfiguration
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Create adaptive geometric hashing grid based on current distribution
        # Spatial hashing with dynamic perturbation: scale by radius to enable differential movement
        max_radius = np.max(radii)
        spatial_hash = np.random.rand(n, 2) * 0.08 * (radii / max_radius)
        
        # Update centers with perturbed spatial hashing
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration using existing constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion with topology reordering
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate distances and find least constrained circle with spatial hashing
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by minimizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Least constrained by smallest min_dists
        
        # Calculate potential growth using current distribution and topology analysis
        current_total = np.sum(radii)
        max_possible_radius = min(0.5 - 1e-3, 1 - 2 * np.max(centers, axis=0))
        
        # Introduce topological constraint: swap with a circle in a different cluster
        # First identify clusters using Voronoi tessellation
        from scipy.spatial import Voronoi
        if n == 26:
            vornoi = Voronoi(centers)
            regions = vornoi.regions
            regions = [r for r in regions if len(r) >= 3]
            region_idx = [vornoi.point_region[i] for i in range(n)]
            cluster_centers = [centers[i] for i in range(n)]
            
            # Group centers into clusters based on Voronoi regions
            clusters = [[] for _ in range(len(regions))]
            for i, r in enumerate(regions):
                clusters[region_idx[i]].append(i)
            
            # Find clusters with minimal current radius and maximal potential expansion
            cluster_radii = [np.mean(radii[i] for i in cl) for cl in clusters]
            target_cluster = np.argmin(cluster_radii)
            
            # Find circle in target cluster that is least constrained (least distance)
            cluster_least = np.argmin([min_dists[i] for i in clusters[target_cluster]])
            cluster_idx = clusters[target_cluster][cluster_least]
            cluster_center = centers[cluster_idx]
            
            # Select topological neighbor in a different cluster for swap
            neighbor_idx = -1
            for i in range(n):
                if i not in clusters[target_cluster] and np.linalg.norm(centers[i] - cluster_center) < 0.2:
                    neighbor_idx = i
                    break
            if neighbor_idx != -1:
                # Swap cluster indices to reconfigure topology
                cluster_idx, neighbor_idx = neighbor_idx, cluster_idx
                temp = v[3*cluster_idx], v[3*cluster_idx+1], v[3*cluster_idx+2]
                v[3*cluster_idx], v[3*cluster_idx+1], v[3*cluster_idx+2] = v[3*neighbor_idx], v[3*neighbor_idx+1], v[3*neighbor_idx+2]
                v[3*neighbor_idx], v[3*neighbor_idx+1], v[3*neighbor_idx+2] = temp
        
        # Apply radius expansion with directional bias and topology reordering adjustment
        total_sum = np.sum(radii)
        expansion_max = 0.009
        expansion_factor = expansion_max / (n - 1)
        
        # Create expansion vector with directed radius expansion based on spatial hashing
        directional_hash = np.random.rand(n, 2) * 0.03
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Directional expansion based on spatial hashing and topology adjustment
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                expansion = expansion_factor * (1.0 + directional_hash[i, 0] * 0.25)
                if adj_weight < 0.15:  # Boost for nearby circles
                    expansion *= 1.15
                new_radii[i] += expansion
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())