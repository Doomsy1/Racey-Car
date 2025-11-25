"""PID controller for autonomous driving."""

import numpy as np


class PIDController:
    """PID controller for steering and speed control."""
    
    def __init__(self, kp, ki, kd, output_limit=None):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limit: Maximum absolute output value (None for no limit)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        
        self.integral = 0.0
        self.previous_error = 0.0
        
    def update(self, error, dt=0.033):
        """
        Update PID controller with new error.
        
        Args:
            error: Current error signal
            dt: Time step (seconds)
            
        Returns:
            Control output
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.previous_error) / dt if dt > 0 else 0.0
        self.previous_error = error
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Apply output limit
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)
        
        return output
    
    def reset(self):
        """Reset integral and previous error."""
        self.integral = 0.0
        self.previous_error = 0.0


class SteeringPIDController:
    """PID controller specifically for steering control."""
    
    def __init__(self, config):
        """
        Initialize steering PID controller from config.
        
        Args:
            config: Configuration dict with steering_pid section
        """
        pid_config = config.get('steering_pid', {})
        kp = pid_config.get('kp', 0.01)
        ki = pid_config.get('ki', 0.0)
        kd = pid_config.get('kd', 0.005)
        
        # Output limit is max angular velocity
        max_angular_velocity = config.get('max_angular_velocity', 2.0)
        
        self.pid = PIDController(kp, ki, kd, output_limit=max_angular_velocity)
        self.image_width = config.get('image_width', 960)
        
    def compute_steering_command(self, centerline_x, dt=0.033):
        """
        Compute steering command from centerline position.
        
        Args:
            centerline_x: X-coordinate of centerline in image (pixels)
            dt: Time step (seconds)
            
        Returns:
            Angular velocity command (rad/s)
        """
        # Compute error: positive if centerline is to the right of image center
        image_center = self.image_width / 2.0
        error = centerline_x - image_center
        
        # Normalize error to [-1, 1] range (optional, helps with tuning)
        normalized_error = error / (self.image_width / 2.0)
        
        # Compute PID output
        angular_velocity = self.pid.update(normalized_error, dt)
        
        return angular_velocity
    
    def reset(self):
        """Reset PID state."""
        self.pid.reset()

