"""Hill obstacles module - obstacles that create elevation/terrain changes."""

import numpy as np
import pybullet as p


def spawn_hill_obstacles(track, physics_client, config=None):
    """
    Spawn obstacles that create hills/elevation changes in the terrain.
    
    Args:
        track: Track object with inner_points and outer_points
        physics_client: PyBullet physics client ID
        config: Optional configuration dict
        
    Returns:
        List of hill obstacle body IDs
    """
    if config is None:
        config = {}
    
    enabled = config.get('enabled', False)
    if not enabled:
        return []
    
    num_hills = config.get('count', 5)
    hill_type = config.get('type', 'mound')
    size_range = config.get('size_range', [0.5, 1.5])
    height_range = config.get('height_range', [0.1, 0.3])
    color = config.get('color', [0.5, 0.4, 0.3, 1.0])
    placement = config.get('placement', 'random')
    
    inner_points = track.inner_points
    outer_points = track.outer_points
    centerline_points = (inner_points + outer_points) / 2.0
    
    rng = track._rng if hasattr(track, '_rng') else np.random.default_rng()
    
    positions = _generate_hill_positions(
        inner_points, outer_points, centerline_points,
        num_hills, placement, rng
    )
    
    hill_ids = []
    
    for pos in positions:
        size = rng.uniform(size_range[0], size_range[1])
        height = rng.uniform(height_range[0], height_range[1])
        
        if hill_type == 'ramp':
            hill_id = _create_ramp(pos, size, height, color, physics_client, rng)
        elif hill_type == 'bump':
            hill_id = _create_bump(pos, size, height, color, physics_client)
        elif hill_type == 'mound':
            hill_id = _create_mound(pos, size, height, color, physics_client)
        else:
            hill_id = _create_mound(pos, size, height, color, physics_client)
        
        if hill_id is not None:
            hill_ids.append(hill_id)
    
    return hill_ids


def _generate_hill_positions(inner_points, outer_points, centerline_points,
                            num_hills, placement, rng):
    """Generate hill positions within track boundaries."""
    positions = []
    
    if placement == 'centerline':
        indices = rng.choice(len(centerline_points), size=min(num_hills, len(centerline_points)), replace=False)
        for idx in indices:
            pos = centerline_points[idx].copy()
            positions.append(pos)
    elif placement == 'sides':
        for _ in range(num_hills):
            segment_idx = rng.integers(0, len(inner_points))
            inner_pos = inner_points[segment_idx]
            outer_pos = outer_points[segment_idx]
            t = rng.uniform(0.2, 0.8)
            pos = inner_pos + t * (outer_pos - inner_pos)
            positions.append(pos)
    else:  # random
        for _ in range(num_hills):
            segment_idx = rng.integers(0, len(inner_points))
            inner_pos = inner_points[segment_idx]
            outer_pos = outer_points[segment_idx]
            t = rng.uniform(0.1, 0.9)
            pos = inner_pos + t * (outer_pos - inner_pos)
            positions.append(pos)
    
    return positions


def _create_ramp(position, size, height, color, physics_client, rng):
    """Create a ramp obstacle (sloped box)."""
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[size/2, size/2, height/2],
        physicsClientId=physics_client
    )
    
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[size/2, size/2, height/2],
        rgbaColor=color,
        physicsClientId=physics_client
    )
    
    # Random rotation around Z-axis
    angle = rng.uniform(0, 2 * np.pi)
    quaternion_z = [0, 0, np.sin(angle/2), np.cos(angle/2)]
    
    # Tilt the ramp (rotate around Y-axis by 30 degrees)
    tilt_angle = np.pi / 6
    tilt_quat_y = np.sin(tilt_angle/2)
    tilt_quat_w = np.cos(tilt_angle/2)
    
    # Combine rotations manually using quaternion multiplication
    # q1 * q2 = (w1*w2 - x1*x2 - y1*y2 - z1*z2,
    #            w1*x2 + x1*w2 + y1*z2 - z1*y2,
    #            w1*y2 - x1*z2 + y1*w2 + z1*x2,
    #            w1*z2 + x1*y2 - y1*x2 + z1*w2)
    q1 = quaternion_z  # [x, y, z, w]
    q2 = [0, tilt_quat_y, 0, tilt_quat_w]
    
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
    
    final_quat = [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
        w1*w2 - x1*x2 - y1*y2 - z1*z2   # w
    ]
    
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position.tolist(),
        baseOrientation=final_quat,
        physicsClientId=physics_client
    )
    
    return body_id


def _create_bump(position, size, height, color, physics_client):
    """Create a bump obstacle (cylinder)."""
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=size / 2.0,
        height=height,
        physicsClientId=physics_client
    )
    
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=size / 2.0,
        length=height,
        rgbaColor=color,
        physicsClientId=physics_client
    )
    
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position.tolist(),
        baseOrientation=[0, 0, 0, 1],
        physicsClientId=physics_client
    )
    
    return body_id


def _create_mound(position, size, height, color, physics_client):
    """Create a mound obstacle (tall cylinder)."""
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=size / 2.0,
        height=height,
        physicsClientId=physics_client
    )
    
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=size / 2.0,
        length=height,
        rgbaColor=color,
        physicsClientId=physics_client
    )
    
    pos_with_height = position.copy()
    pos_with_height[2] += height / 2.0
    
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=pos_with_height.tolist(),
        baseOrientation=[0, 0, 0, 1],
        physicsClientId=physics_client
    )
    
    return body_id

