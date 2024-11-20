import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from vpython import vector, canvas, rate
from particle import Particle

class System():
    """System of particles."""
    def __init__(self, particles):
        self.particles = particles
        self.animation = self.particles[0].animation
        self.particle_dict = {p.name:p for p in particles}
        self.positions = {particle.name:[] for particle in particles}
        self.kinetic_energy = []
        self.potential_energy = []
        self.angular_momentum = []

    def compute_forces(self):
        """Non parallel method of computing forces."""

        if self.animation:
            all_accel = {name:vector(0,0,0) for name in self.particle_dict}
        else:
            all_accel = {name:np.zeros(3) for name in self.particle_dict}

        for i, p1 in enumerate(self.particles):
            for p2 in self.particles[i+1:]:
                accel_multiplier = p1.get_accel_multiplier(p2)
                all_accel[p1.name] += accel_multiplier*p2.mass
                all_accel[p2.name] -= accel_multiplier*p1.mass

        for name in all_accel:
            self.particle_dict[name].acceleration = all_accel[name]

    def compute_force_i(self, args):
        """Compute force between two particles."""
        p1, p2 = args
        accel_multiplier = p1.get_accel_multiplier(p2)
        return (p1, accel_multiplier*p2.mass), (p2, -accel_multiplier*p1.mass)

    def compute_PE_i(self, args):
        """Compute potential energy between two particles."""
        p1, p2 = args
        return p1.get_PE(p2)

    def compute_chunks(self, args):
        """Compute for a chunk."""
        compute_func, chunk = args
        results = []
        for p1, p2 in chunk:
            results.append(compute_func((p1,p2)))
        return results

    def compute_parallel(self, compute_func):
        """Performs all pairwise computations in parallel."""
        tasks = []
        for i, p1 in enumerate(self.particles):
            for p2 in self.particles[i+1:]:
                tasks.append((p1,p2))

        cpu_count = multiprocessing.cpu_count()
        num_chunks = 2*cpu_count
        chunk_size = len(tasks)//num_chunks
        chunks = [tasks[i:i+chunk_size] for i in range(0, len(tasks), chunk_size)]

        with multiprocessing.Pool(processes=cpu_count) as pool:
            results = pool.map(self.compute_chunks, [(compute_func, chunk) for chunk in chunks])
        return results

    def compute_forces_parallel(self):
        """Compute forces using parallelism."""
        results = self.compute_parallel(self.compute_force_i)

        for p in self.particles:
            p.acceleration = np.zeros(3)

        for result_chunk in results:
            for _, result in result_chunk:
                result[0].acceleration += result[1]

    def compute_PE_parallel(self):
        """Compute potential energy using parallelism."""
        results = self.compute_parallel(self.compute_PE_i)

        total_PE = 0
        for result_chunk in results:
            for result in result_chunk:
                total_PE += result

        self.potential_energy.append(total_PE)


    def update(self, dt, parallel=False):
        """Updates system by one time step."""
        if parallel:
            self.compute_forces_parallel()
        else:
            self.compute_forces()
        for p in self.particles:
            p.integrate(dt)

        for p in self.particles:
            if self.animation:
                self.positions[p.name].append(vector(p.position.x,
                                                     p.position.y,
                                                     p.position.z))
            else:
                self.positions[p.name].append(p.position.copy())

        if not self.animation:
            self.kinetic_energy.append(sum(p.get_KE() for p in self.particles))
            self.angular_momentum.append(sum(p.get_L() for p in self.particles))

            if parallel:
                self.compute_PE_parallel()
            else:
                potential = 0
                for i, p1 in enumerate(self.particles):
                    for p2 in self.particles[i+1:]:
                        potential += p1.get_PE(p2)
                self.potential_energy.append(potential)


    def simulate(self, dt, num_steps, parallel=False):
        """Runs the simulation."""
        for _ in range(num_steps):
            self.update(dt, parallel)


def generate_particles(n, animation=False): # 3D
    """
    Generates a list of particles for the computation time vs. N graph

    n: number of particles
    """
    list_of_particles = []

    p_lim = 1e11
    v_lim = 2000

    for i in range(n):

        mass = np.random.uniform(1, 9) * 10 ** np.random.uniform(26, 28)
        p = np.array([np.random.uniform(-p_lim, p_lim),
                    np.random.uniform(-p_lim, p_lim),
                    np.random.uniform(-p_lim, p_lim)])
        v = np.array([np.random.uniform(-v_lim, v_lim),
                    np.random.uniform(-v_lim, v_lim),
                    np.random.uniform(-v_lim, v_lim)])
        name = str(i)

        list_of_particles.append(Particle(mass, p, v, name, animation))

    return list_of_particles

def create_animation(all_particles):
    """Creates animation from given particles."""
    scene = canvas()
    scene.center = vector(0, 0, 0)

    s = System(all_particles)
    dt = 3000
    while True:
        rate(3000)
        s.update(dt)

def main():
    np.random.seed(2)

    n = 5
    all_particles = generate_particles(n)

    dt = 30000
    num_steps = 365*24*60*60 // dt

    s = System(all_particles)
    s.simulate(dt, num_steps, n>500)

    plt.figure()
    for p in s.positions.values():
        plt.plot([i[0] for i in p], [i[1] for i in p], "-k")
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()


    plt.figure()
    plt.plot(np.linspace(0, dt*num_steps, num_steps), s.kinetic_energy, label="KE")
    plt.plot(np.linspace(0, dt*num_steps, num_steps), s.potential_energy, label="PE")
    plt.plot(np.linspace(0, dt*num_steps, num_steps), 
             np.array(s.kinetic_energy)+np.array(s.potential_energy), label="Total Energy")
    plt.legend(loc="upper left")
    plt.xlabel(r"Time")
    plt.ylabel(r"Energy (J)")
    plt.title('Energy is conserved')
    plt.show()

    plt.figure()
    plt.plot(np.linspace(0, dt*num_steps, num_steps), [i[0] for i in s.angular_momentum], label=r"$L_x$")
    plt.plot(np.linspace(0, dt*num_steps, num_steps), [i[1] for i in s.angular_momentum], label=r"$L_y$")
    plt.plot(np.linspace(0, dt*num_steps, num_steps), [i[2] for i in s.angular_momentum], label=r"$L_z$")
    plt.legend(loc="upper left")
    plt.xlabel(r"Time")
    plt.ylabel(r"Angular momentum (kg $\frac{\text{m}^2}{\text{s}})$")
    plt.title('Angular momentum is conserved')
    plt.show()

    all_particles = generate_particles(n, animation=True)
    create_animation(all_particles)

if __name__ == "__main__":
    main()
    
