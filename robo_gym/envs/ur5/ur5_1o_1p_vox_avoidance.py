#!/usr/bin/env python3
from copy import deepcopy
import math, copy
import numpy as np
from scipy.spatial.transform import Rotation as R
import gym
from gym import spaces
from gym.utils import seeding
from robo_gym.utils import utils, ur_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2

from robo_gym.envs.ur5.ur5_avoidance import MovingBoxTargetUR5

DEBUG = True

class ObstacleAvoidance1Box1PointsVoxelOccupancyUR5(MovingBoxTargetUR5):

    def _get_observation_space(self):
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """

        # Joint position range tolerance
        pos_tolerance = np.full(6,0.1)
        # Joint positions range used to determine if there is an error in the sensor readings
        max_joint_positions = np.add(np.full(6, 1.0), pos_tolerance)
        min_joint_positions = np.subtract(np.full(6, -1.0), pos_tolerance)
        # Target coordinates range
        target_range = np.full(3, np.inf)
        # Joint positions range tolerance
        vel_tolerance = np.full(6,0.5)

        max_delta_start_positions = np.add(np.full(6, 1.0), pos_tolerance)
        min_delta_start_positions = np.subtract(np.full(6, -1.0), pos_tolerance)
        
        # Target coordinates (with respect to forearm frame) range
        target_forearm_range = np.full(3, np.inf)

        # voxel occupancy 
        max_voxel_occupancy = np.full(16, 1) # 1 ... occupied
        min_voxel_occupancy = np.full(16, 0) # 0 ... not occupied

        # Definition of environment observation_space
        max_obs = np.concatenate(( max_joint_positions, max_delta_start_positions, max_voxel_occupancy))
        min_obs = np.concatenate(( min_joint_positions, min_delta_start_positions, min_voxel_occupancy))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
    
    def reset(self, initial_joint_positions = None, type='random'):
        """Environment reset.

        Args:
            initial_joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())
        
        # NOTE: maybe we can find a cleaner version when we have the final envs (we could prob remove it for the avoidance task altogether)
        # Set initial robot joint positions
        if initial_joint_positions:
            assert len(initial_joint_positions) == 6
            ur_initial_joint_positions = initial_joint_positions
        elif (len(self.last_position_on_success) != 0) and (type=='continue'):
            ur_initial_joint_positions = self.last_position_on_success
        else:
            ur_initial_joint_positions = self._get_initial_joint_positions()

        rs_state[6:12] = self.ur._ur_joint_list_to_ros_joint_list(ur_initial_joint_positions)


        # Set initial state of the Robot Server
        
        string_params = {"object_0_function": "3d_spline"}

        r = np.random.uniform()

        if r <= 0.75:
            # object in front of the robot
            float_params = {"object_0_x_min": -0.7, "object_0_x_max": 0.7, "object_0_y_min": 0.2, "object_0_y_max": 1.0, \
                            "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                            "n_sampling_points": 4000}
        elif r <= 0.83:
            # object behind robot
            float_params = {"object_0_x_min": -0.7, "object_0_x_max": 0.7, "object_0_y_min": - 0.7, "object_0_y_max": -0.2, \
                            "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                            "n_sampling_points": 4000}
        elif r <= 0.91:
            # object on the left side of the  robot
            float_params = {"object_0_x_min": 0.3, "object_0_x_max": 0.7, "object_0_y_min": - 0.7, "object_0_y_max": 0.7, \
                            "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                            "n_sampling_points": 4000}
        else :
            # object on the right side of the  robot
            float_params = {"object_0_x_min": -0.2, "object_0_x_max": -0.7, "object_0_y_min": - 0.7, "object_0_y_max": 0.7, \
                            "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                            "n_sampling_points": 4000}
        
        state_msg = robot_server_pb2.State(state = rs_state.tolist(), float_params = float_params, string_params = string_params)
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))
        self.prev_rs_state = copy.deepcopy(rs_state)

        # Check if the length of the Robot Server state received is correct
        if not len(rs_state)== self._get_robot_server_state_len():
            raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # save start position
        self.start_position = self.state[3:9]

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()
        
        # check if current position is in the range of the initial joint positions
        if (len(self.last_position_on_success) == 0) or (type=='random'):
            joint_positions = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
            tolerance = 0.1
            for joint in range(len(joint_positions)):
                if (joint_positions[joint]+tolerance < self.initial_joint_positions_low[joint]) or  (joint_positions[joint]-tolerance  > self.initial_joint_positions_high[joint]):
                    raise InvalidStateError('Reset joint positions are not within defined range')
        
        num_voxel_x = 4
        num_voxel_y = 2
        num_voxel_z = 2
        voxel_size = 0.5
        voxel_centers_in_occupancy_frame = (voxel_size/2.0 + voxel_size * np.mgrid[0:num_voxel_x,0:num_voxel_y,0:num_voxel_z]).reshape(3,-1).T
        if self.real_robot:
            # make sure this transform matches the static_transform_publisher in the robot server (i.e. in the launch file)
            quaternion_occupancy_to_base = R.from_euler('z', -2.356).as_quat()
            translation_occupancy_to_base = np.array([0.706, 0.706, 0.0])
            self.voxel_centers_in_reference_frame = utils.change_reference_frame(voxel_centers_in_occupancy_frame, translation_occupancy_to_base, quaternion_occupancy_to_base)
        else:
            # 
            quaternion_occupancy_to_world = R.identity().as_quat()
            translation_occupancy_to_world = np.array([-1.0, 0.0, 0.1])
            self.voxel_centers_in_reference_frame = utils.change_reference_frame(voxel_centers_in_occupancy_frame, translation_occupancy_to_world, quaternion_occupancy_to_world)
        

        # occ frame [1.25, 0.25, 0.25] -> base [0, -0.35, 0.25]
        if DEBUG:
            for i, v in enumerate(self.voxel_centers_in_reference_frame):
                print("voxel in base_frame ", i, v, voxel_centers_in_occupancy_frame[i])

        return self.state
    
    def _robot_server_state_to_env_state(self, rs_state):
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        # Convert to numpy array and remove NaN values
        rs_state = np.nan_to_num(np.array(rs_state))

        # Transform cartesian coordinates of target to polar coordinates 
        # with respect to the end effector frame
        target_coord = rs_state[0:3]

        # Transform joint positions and joint velocities from ROS indexing to
        # standard indexing
        ur_j_pos = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
        ur_j_vel = self.ur._ros_joint_list_to_ur_joint_list(rs_state[12:18])

        # Normalize joint position values
        ur_j_pos_norm = self.ur.normalize_joint_values(joints=ur_j_pos)

        # start joint positions
        start_joints = self.ur.normalize_joint_values(self._get_initial_joint_positions())
        delta_joints = ur_j_pos_norm - start_joints

        # Transform cartesian coordinates of target to polar coordinates 
        # with respect to the forearm

        voxel_occupancy = rs_state[33:49]
        # Compose environment state
        state = np.concatenate((ur_j_pos_norm, delta_joints, voxel_occupancy))

        return state
    
    def print_state_action_info(self, rs_state, action):
        env_state = self._robot_server_state_to_env_state(rs_state)

        print('Action:', action)
        print('Last A:', self.last_action)
        print('Distance (target 1): {:.2f}'.format(env_state[0]))
        print('Polar 1 (degree): {:.2f}'.format(env_state[1] * 180/math.pi))
        print('Polar 2 (degree): {:.2f}'.format(env_state[2] * 180/math.pi))
        print('Distance (target 2): {:.2f}'.format(env_state[15]))
        print('Polar 1 (degree): {:.2f}'.format(env_state[16] * 180/math.pi))
        print('Polar 2 (degree): {:.2f}'.format(env_state[17] * 180/math.pi))
        print('Joint Positions: [1]:{:.2e} [2]:{:.2e} [3]:{:.2e} [4]:{:.2e} [5]:{:.2e} [6]:{:.2e}'.format(*env_state[3:9]))
        print('Joint PosDeltas: [1]:{:.2e} [2]:{:.2e} [3]:{:.2e} [4]:{:.2e} [5]:{:.2e} [6]:{:.2e}'.format(*env_state[9:15]))
        print('Sum of Deltas: {:.2e}'.format(sum(abs(env_state[9:15]))))
        print()

    def _reward(self, rs_state, action):
        # TODO coordinates of the 2 objects with respect to the forearm link should be integrated
        # TODO: remove print when not needed anymore
        # print('action', action)
        env_state = self._robot_server_state_to_env_state(rs_state)

        reward = 0
        done = False
        info = {}

        # minimum and maximum distance the robot should keep to the obstacle
        minimum_distance = 0.45 # m
        maximum_distance = 0.6 # m

        # calculate distance to the target
        target_coord = np.array(rs_state[0:3])
        ee_coord = np.array(rs_state[18:21])
        distance_to_target = np.linalg.norm(target_coord - ee_coord)   

        distance_to_target_2 = env_state[-3]
        
        ee_to_base_translation = np.array(rs_state[18:21])
        ee_to_base_quaternion = np.array(rs_state[21:25])
        ee_to_base_rotation = R.from_quat(ee_to_base_quaternion)
        base_to_ee_rotation = ee_to_base_rotation.inv()
        # to invert the homogeneous transformation
        # R' = R^-1
        base_to_ee_quaternion = base_to_ee_rotation.as_quat()
        # t' = - R^-1 * t
        base_to_ee_translation = -base_to_ee_rotation.apply(ee_to_base_translation)

        if DEBUG:
            print('base_to_ee_translation', base_to_ee_translation)
            print('base_to_ee_quaternion', base_to_ee_quaternion)
        voxel_ee_distance = []
        for voxel_index, voxel_center in enumerate(self.voxel_centers_in_reference_frame):
            voxel_center_ee_frame = utils.change_reference_frame(voxel_center, base_to_ee_translation, base_to_ee_quaternion)
            if DEBUG:
                print(voxel_index, voxel_center_ee_frame, np.linalg.norm(voxel_center_ee_frame))
            voxel_ee_distance.append(np.linalg.norm(voxel_center_ee_frame))


        voxels = env_state[12:27]
        print('voxels', voxels)
        print('vox_dist', voxel_ee_distance)
        # if DEBUG: 
        #     print('Voxels:', voxels, 'SumVox:', sum(voxels))
        #     assert sum(voxels) <= 1

        # voxels = np.array(voxels)
        # voxel_index = voxels.where(voxels == 1)

        # distance_to_target = voxel_ee_distance[voxel_index]

        print('distance to target', distance_to_target)
    
        # ! not used yet
        # ? we could train the robot to always "look" at the target just because it would look cool
        polar_1 = abs(env_state[1] * 180/math.pi)
        polar_2 = abs(env_state[2] * 180/math.pi)
        # reward polar coords close to zero
        # polar 1 weighted by max_steps
        p1_r = (1 - (polar_1 / 90)) * (1/1000)
        # if p1_r <= 0.001:
        #     reward += p1_r
        
        # polar 1 weighted by max_steps
        p2_r = (1 - (polar_2 / 90)) * (1/1000)
        # if p2_r <= 0.001:
        #     reward += p2_r

        # TODO: depends on the stating pos being in the state, maybe find better solution
        # difference in joint position current vs. starting position
        # delta_joint_pos = env_state[3:9] - env_state[-6:]
        delta_joint_pos = env_state[9:15]

        # reward for being in the defined interval of minimum_distance and maximum_distance
        dr = 0
        if abs(env_state[-6:]).sum() < 0.5:
            dr = 2 * (1 - (abs(delta_joint_pos).sum()/0.5)) * (1/1000)
            reward += dr
        
        # reward moving as less as possible
        act_r = 0 
        if abs(action).sum() <= action.size:
            act_r = 1 * (1 - (np.square(action).sum()/action.size)) * (1/1000)
            reward += act_r

        # punish big deltas in action
        act_delta = 0
        for i in range(len(action)):
            if abs(action[i] - self.last_action[i]) > 0.5:
                a_r = - 0.2 * (1/1000)
                act_delta += a_r
                reward += a_r
        
        # ? First try is to just split the distance reward
        # punish if the obstacle gets too close
        dist_1 = 0 
        dist_2 = 0
        if (distance_to_target < minimum_distance) or (distance_to_target_2 < minimum_distance):
            dist_1 = -2 * (1/self.max_episode_steps) # -2
            reward += dist_1


        # punish if the robot moves too far away from the obstacle
        dist_max = 0
        if distance_to_target > maximum_distance:
            dist_max = -1 * (1/self.max_episode_steps)
            #reward += dist_max

        # TODO: we could remove this if we do not need to punish failure or reward success
        # Check if robot is in collision
        collision = True if rs_state[25] == 1 else False
        if collision:
            # reward = -5
            done = True
            info['final_status'] = 'collision'
            info['target_coord'] = target_coord
            self.last_position_on_success = []

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'
            info['target_coord'] = target_coord
            self.last_position_on_success = []
        

        if DEBUG: self.print_state_action_info(rs_state, action)
        # ? DEBUG PRINT
        if DEBUG: print('reward composition:', 'dr =', round(dr, 5), 'no_act =', round(act_r, 5), 'min_dist_1 =', round(dist_1, 5), 'min_dist_2 =', round(dist_2, 5), 'delta_act', round(act_delta, 5))


        return reward, done, info

    def _get_robot_server_state_len(self):
        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.

        Returns:
            int: Length of the Robot Server state.

        """

        target = [0.0]*6
        ur_j_pos = [0.0]*6
        ur_j_vel = [0.0]*6
        ee_to_base_transform = [0.0]*7
        ur_collision = [0.0]
        forearm_to_base_transform = [0.0]*7
        voxel_occupancy = [0.0]*16
        rs_state = target + ur_j_pos + ur_j_vel + ee_to_base_transform + ur_collision + forearm_to_base_transform + voxel_occupancy
        

        return len(rs_state)

class ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF3(ObstacleAvoidance1Box1PointsVoxelOccupancyUR5):
    def _get_action_space(self):
        return spaces.Box(low=np.full((3), -1.0), high=np.full((3), 1.0), dtype=np.float32)

class ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF5(ObstacleAvoidance1Box1PointsVoxelOccupancyUR5):
    def _get_action_space(self):
        return spaces.Box(low=np.full((5), -1.0), high=np.full((5), 1.0), dtype=np.float32)


class ObstacleAvoidance1Box1PointsVoxelOccupancyUR5Sim(ObstacleAvoidance1Box1PointsVoxelOccupancyUR5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=-0.78\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=true \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=1moving1point_2_2_4_voxel \
        n_objects:=1.0 \
        object_0_model_name:=box100 \
        object_0_frame:=target \
        camera1_gazebo:=true \
        camera1_link_y:=1.5 \
        camera1_link_z:=2.0 \
        camera1_link_pitch:=1.0 \
        camera1_link_yaw:=-1.5 \
        use_voxel_occupancy:=True \
        rviz_config_file:=ur_occupancy_rl.rviz"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        ObstacleAvoidance1Box1PointsVoxelOccupancyUR5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ObstacleAvoidance1Box1PointsVoxelOccupancyUR5Rob(ObstacleAvoidance1Box1PointsVoxelOccupancyUR5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=1moving2points n_objects:=1.0 object_0_frame:=target

class ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF3Sim(ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF3, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=-0.78\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=true \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=1moving1point_2_2_4_voxel \
        n_objects:=1.0 \
        object_0_model_name:=box100 \
        object_0_frame:=target \
        camera1_gazebo:=True \
        camera1_link_y:=1.5 \
        camera1_link_z:=2.0 \
        camera1_link_pitch:=1.0 \
        camera1_link_yaw:=-1.5 \
        use_voxel_occupancy:=True \
        rviz_config_file:=ur_occupancy_rl.rviz"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF3.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF3Rob(ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF3):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=1moving2points n_objects:=1.0 object_0_frame:=target

class ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF5Sim(ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=-0.78\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=true \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=1moving1point_2_2_4_voxel \
        n_objects:=1.0 \
        object_0_model_name:=box100 \
        object_0_frame:=target \
        camera1_gazebo:=true \
        camera1_link_y:=1.5 \
        camera1_link_z:=2.0 \
        camera1_link_pitch:=1.0 \
        camera1_link_yaw:=-1.5 \
        use_voxel_occupancy:=True \
        rviz_config_file:=ur_occupancy_rl.rviz"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF5Rob(ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=1moving2points n_objects:=1.0 object_0_frame:=target

# maybe later
# roslaunch ur_robot_server ur5_sim_robot_server.launch camera1_gazebo:=true gui:=true camera1_link_y:=1 camera1_link_z:=1 camera1_link_pitch:=1.0 camera1_link_yaw:=-1.5 gazebo_gui:=True use_voxel_occupancy:=True 