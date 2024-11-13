import numpy as np
from vpython import sphere, vector, color, mag

G = 6.6742e-11

class Particle:
    """Particle for N Body Simulations."""
    def __init__(self, mass, position, velocity, name, animation=False):
        self.mass = mass
        self.animation = animation
        if animation:
            self.position = vector(*position)
            self.velocity = vector(*velocity)
            self.acceleration = vector(0,0,0)
            self.sphere = sphere(pos=self.position, radius=1e10, color=color.blue)
        else:
            self.position = position
            self.velocity = velocity
            self.acceleration = np.zeros(3)
        self.name = name



    def integrate(self, dt):
        """
        Integrates all velocities and positions using leapfrog method.
        """
        self.velocity += self.acceleration * dt/2
        self.position += self.velocity * dt
        self.velocity += self.acceleration * dt/2
        if self.animation:
            self.sphere.pos = self.position
            speed = self.velocity.mag
            max_speed = 6000
            normalized_speed = min(speed/max_speed, 1.0)
            self.sphere.color = vector(normalized_speed, 0, 1-normalized_speed)

    def get_accel_multiplier(self, other):
        """
        Gets the "acceleration multiplier" between two particles: G/r^2
        """
        # vector from self to other
        r = other.position - self.position
        if self.animation:
            distance = mag(r)
        else:
            distance = np.linalg.norm(r)
        return G * r / (distance**3)

    def get_KE(self):
        """Returns kinetic energy of particle."""
        return 1/2 * self.mass * np.linalg.norm(self.velocity)**2

    def get_PE(self, other):
        """Returns potential energy of particle relative to another."""
        return -G * self.mass * other.mass / (np.linalg.norm(other.position-self.position))

    def get_L(self):
        """Returns angular momenutum of particle relative to origin."""
        return self.mass * np.cross(self.position, self.velocity)
