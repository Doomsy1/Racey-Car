"""Friction obstacles module - obstacles that cause friction when car contacts them."""

import numpy as np
import pybullet as p


def spawn_track_friction_surface(track, physics_client, car_id, config=None):
    """
    Spawn a friction surface that spans the entire track area (fills inside track barriers).
    
    Args:
        track: Track object with inner_points and outer_points
        physics_client: PyBullet physics client ID
        car_id: Car body ID for friction application
        config: Optional configuration dict
        
    Returns:
        List of surface segment body IDs
    """
    if config is None:
        config = {}
    
    enabled = config.get('enabled', False)
    if not enabled:
        return []
    
    friction_coefficient = config.get('friction_coefficient', 0.8)
    height = config.get('height', 0.001)  # Thickness of the surface (very thin)
    color = config.get('color', [0.3, 0.3, 0.3, 1.0])  # Default gray
    
    inner_points = track.inner_points
    outer_points = track.outer_points
    num_segments = len(inner_points)
    
    surface_ids = []
    
    # Create segments that fill the track area
    for i in range(num_segments):
        # Get current and next points (wrap around for closed loop)
        inner_curr = inner_points[i]
        inner_next = inner_points[(i + 1) % num_segments]
        outer_curr = outer_points[i]
        outer_next = outer_points[(i + 1) % num_segments]
        
        # Create two triangles per segment to form a quad
        # Triangle 1: inner_curr, outer_curr, inner_next
        # Triangle 2: outer_curr, outer_next, inner_next
        
        # Calculate center and dimensions for a box that covers this segment
        # Use the average of the four corner points
        center = (inner_curr + inner_next + outer_curr + outer_next) / 4.0
        
        # Calculate segment length (along track)
        seg_length = np.linalg.norm((inner_next + outer_next) / 2.0 - (inner_curr + outer_curr) / 2.0)
        
        # Calculate track width at this segment
        track_width = np.linalg.norm(outer_curr - inner_curr)
        
        # Create a box that spans this segment
        # Position it so the top surface is at ground level (z=0)
        # Box center is at -height/2 so top is at z=0, bottom is at z=-height
        # The very thin height (0.001m) ensures it doesn't catch the car
        box_position = [center[0], center[1], -height / 2.0]
        
        # Calculate orientation: align box along track direction
        track_direction = ((inner_next + outer_next) / 2.0) - ((inner_curr + outer_curr) / 2.0)
        track_direction[2] = 0  # Keep in XY plane
        track_direction = track_direction / (np.linalg.norm(track_direction) + 1e-6)
        
        # Calculate angle for rotation
        angle = np.arctan2(track_direction[1], track_direction[0])
        
        # Create quaternion for rotation around Z-axis
        quaternion = [0, 0, np.sin(angle / 2.0), np.cos(angle / 2.0)]
        
        # Create collision and visual shapes
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[seg_length / 2.0, track_width / 2.0, height / 2.0],
            physicsClientId=physics_client
        )
        
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[seg_length / 2.0, track_width / 2.0, height / 2.0],
            rgbaColor=color,
            physicsClientId=physics_client
        )
        
        # Create the body
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=box_position,
            baseOrientation=quaternion,
            physicsClientId=physics_client
        )
        
        if body_id is not None:
            # Apply friction coefficient
            p.changeDynamics(
                body_id,
                -1,
                lateralFriction=friction_coefficient,
                spinningFriction=friction_coefficient * 0.5,
                rollingFriction=friction_coefficient * 0.1,
                physicsClientId=physics_client
            )
            
            # Ensure car has friction enabled
            p.changeDynamics(
                car_id,
                -1,
                lateralFriction=0.8,
                physicsClientId=physics_client
            )
            
            surface_ids.append(body_id)
    
    return surface_ids


def spawn_friction_obstacles(track, physics_client, car_id, config=None):
    """
    Spawn obstacles with low friction that slow the car on contact.
    
    Args:
        track: Track object with inner_points and outer_points
        physics_client: PyBullet physics client ID
        car_id: Car body ID for friction application
        config: Optional configuration dict
        
    Returns:
        List of obstacle body IDs
    """
    if config is None:
        config = {}
    
    enabled = config.get('enabled', False)
    if not enabled:
        return []
    
    num_obstacles = config.get('count', 50)
    size_range = config.get('size_range', [0.2, 0.4])
    height = config.get('height', 0.01)
    placement = config.get('placement', 'aligned')
    
    # Three colors with friction levels: Yellow (low) -> Orange (medium) -> Red (high)
    friction_configs = [
        {'color': [1.0, 1.0, 0.0, 1.0], 'friction': 0.2},  # Yellow - low friction
        {'color': [1.0, 0.5, 0.0, 1.0], 'friction': 0.5},  # Orange - medium friction
        {'color': [1.0, 0.0, 0.0, 1.0], 'friction': 0.8},  # Red - high friction
    ]
    
    inner_points = track.inner_points
    outer_points = track.outer_points
    
    rng = track._rng if hasattr(track, '_rng') else np.random.default_rng()
    centerline_points = (inner_points + outer_points) / 2.0
    
    positions = _generate_obstacle_positions(
        inner_points, outer_points, centerline_points,
        num_obstacles, placement, rng
    )
    
    obstacle_ids = []
    
    for i, pos in enumerate(positions):
        size = rng.uniform(size_range[0], size_range[1])
        # Cycle through friction levels: yellow -> orange -> red
        friction_config = friction_configs[i % len(friction_configs)]
        color = friction_config['color']
        friction_coefficient = friction_config['friction']
        
        obstacle_id = _create_friction_patch(pos, size, height, color, physics_client, 'box')
        
        if obstacle_id is not None:
            p.changeDynamics(
                obstacle_id,
                -1,
                lateralFriction=friction_coefficient,
                spinningFriction=friction_coefficient * 0.5,
                rollingFriction=friction_coefficient * 0.1,
                physicsClientId=physics_client
            )
            
            p.changeDynamics(
                car_id,
                -1,
                lateralFriction=0.8,
                physicsClientId=physics_client
            )
            
            obstacle_ids.append(obstacle_id)
    
    return obstacle_ids


def _generate_obstacle_positions(inner_points, outer_points, centerline_points,
                                 num_obstacles, placement, rng):
    """Generate obstacle positions within track boundaries."""
    positions = []
    
    if placement == 'aligned':
        # Evenly spaced along centerline with slight lateral variation
        num_centerline = len(centerline_points)
        if num_obstacles <= num_centerline:
            # Evenly sample centerline points
            step = num_centerline / num_obstacles
            for i in range(num_obstacles):
                idx = int(i * step) % num_centerline
                center_pos = centerline_points[idx]
                # Add slight lateral offset (within track width)
                lateral_offset = rng.uniform(-0.3, 0.3)  # Small variation
                # Calculate perpendicular direction
                if idx < num_centerline - 1:
                    direction = centerline_points[(idx + 1) % num_centerline] - centerline_points[idx]
                else:
                    direction = centerline_points[idx] - centerline_points[idx - 1]
                # Perpendicular vector (rotate 90 degrees in XY plane)
                perp = np.array([-direction[1], direction[0], 0])
                perp = perp / (np.linalg.norm(perp) + 1e-6)
                pos = center_pos + perp * lateral_offset
                positions.append(pos)
        else:
            # More obstacles than centerline points - repeat with variations
            for i in range(num_obstacles):
                idx = i % num_centerline
                center_pos = centerline_points[idx]
                lateral_offset = rng.uniform(-0.4, 0.4)
                if idx < num_centerline - 1:
                    direction = centerline_points[(idx + 1) % num_centerline] - centerline_points[idx]
                else:
                    direction = centerline_points[idx] - centerline_points[idx - 1]
                perp = np.array([-direction[1], direction[0], 0])
                perp = perp / (np.linalg.norm(perp) + 1e-6)
                pos = center_pos + perp * lateral_offset
                positions.append(pos)
    elif placement == 'centerline':
        indices = rng.choice(len(centerline_points), size=min(num_obstacles, len(centerline_points)), replace=False)
        for idx in indices:
            pos = centerline_points[idx].copy()
            positions.append(pos)
    elif placement == 'sides':
        for _ in range(num_obstacles):
            segment_idx = rng.integers(0, len(inner_points))
            inner_pos = inner_points[segment_idx]
            outer_pos = outer_points[segment_idx]
            t = rng.uniform(0.2, 0.8)
            pos = inner_pos + t * (outer_pos - inner_pos)
            positions.append(pos)
    else:  # random
        for _ in range(num_obstacles):
            segment_idx = rng.integers(0, len(inner_points))
            inner_pos = inner_points[segment_idx]
            outer_pos = outer_points[segment_idx]
            t = rng.uniform(0.1, 0.9)
            pos = inner_pos + t * (outer_pos - inner_pos)
            positions.append(pos)
    
    return positions


def _create_friction_patch(position, size, height, color, physics_client, shape_type='box'):
    """Create a friction patch obstacle."""
    if shape_type == 'cylinder':
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
    else:  # box
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
    
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position.tolist(),
        baseOrientation=[0, 0, 0, 1],
        physicsClientId=physics_client
    )
    
    return body_id

