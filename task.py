import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = 2 * self.action_repeat
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        self.step_count = 0
        self.max_iteration = 100
        # Goal
        self.target_pos = np.array([0., 0., 10.])
        self.target_vel = np.array([0., 0., 0.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        error_position = np.linalg.norm(self.target_pos[2] - self.sim.pose[2])

        if self.sim.pose[2] > self.target_pos[2]:
            reward = self.target_pos[2]
        else:
            reward = error_position

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        rotor_speeds = np.zeros((4,))+rotor_speeds

        reward = 0
        pose_all = []

        self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities

        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)
            reward = self.get_reward()
            pose_all.append(self.sim.pose[2])
            pose_all.append(self.sim.v[2])

        next_state = np.array(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.step_count = 0
        init_pose = np.array([0., 0., np.random.normal(0.3, 0.1), 0., 0., 0.])
        self.sim = PhysicsSim(init_pose, None, None, 5.)
        l = [self.sim.pose[2], self.sim.v[2]] * self.action_repeat

        state = np.array(l)
        return state
