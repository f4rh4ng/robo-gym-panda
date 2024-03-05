<!-- omit in toc -->
# robo-gym-panda 

<br>

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/robo-gym)


**This Project introduces an extension for the ``robo-gym`` toolkit. It adds the support of Franka Emika Panda Robot.**


Added features :

- integration of **Franka Emika Panda** robot for both **simulated** and **real** environment
- inclusion of two robotic arm manipulation tasks for Franka Emika Panda robot:

  - End-effector positioning task (successfully solved using DRL algorithms in simulation environment and tested on real robot as well.)

    ![ezgif com-gif-maker](https://github.com/f4rh4ng/robo-gym-panda/assets/89604072/d0b6d5a1-9122-4e1b-9ab0-505a37a5a47b)
    ![lessthan50robot-ezgif com-repair](https://github.com/f4rh4ng/robo-gym-panda/assets/89604072/e418f397-3589-4626-bcf3-986fa8d9e976)

  - Task of following a moving target in 3D space (successfully solved using DRL algorithms in simulation environment)

    ![ezgif com-gif-maker(1)](https://github.com/f4rh4ng/robo-gym-panda/assets/89604072/b328a398-50dc-41f2-8683-0f43f21f6085)

# Installation
[back to top](#robo-gym-panda)

<!-- omit in toc -->
## Environment Side
**Requirements:** Python >= 3.6, Pytorch: '1.11.0+cu102' (trained with this version)

You can perform a minimal install of robo-gym-panda with:

```bash
git clone https://github.com/f4rh4ng/robo-gym-panda.git
cd robo-gym
pip install -e .
```

By following the instruction above, the setup for using [D4PG algorithm](https://github.com/schatty/oprl/tree/legacy_d4pg), including the environments for Franka Emika Panda robot, will already be included in ``robo-gym-panda\docs``, so no further installation is needed.

For using algorithms such as DDPG and TD3 we have used the [stable-baseline3](https://github.com/DLR-RM/stable-baselines3) library. For using these algorithms the following installation is required:

```bash
pip3 install stable-baselines3[extra]
```
<!-- omit in toc -->
## Robot Server Side
**Requirements:** Ubuntu 20.04 (recommended) or 18.04.

The Robot Server Side can be installed on the same machine running the Environment Side
and/or on other multiple machines.

Install [robo-gym-robot-servers](https://github.com/f4rh4ng/robo-gym-robot-servers)
following the instructions in the repository's README.

# How to use

<!-- omit in toc -->
## Simulated Environments

Open a terminal and start a server manager using following command (In all of PCs used in network for training):

```sh
start-server-manager
```
**NOTE:** Each time when the task is finished, make sure to run the following command before starting a new server manager and starting another task:

```sh
kill-all-robot-servers
```
After starting the server manager you can start the robot server and train and test different algorithms using following instructions.

<!-- omit in toc -->
### TD3 Algorithm ([stable-baseline3](https://github.com/DLR-RM/stable-baselines3))
#### Training:
Run the following file:
```sh
python docs/examples/stable-baselines/Main.py
```

In **Main.py** you can specify the IP of the machine running the **robot-server**, the number of training steps, the training environment that can be loaded and other settings.

The name of the registered training gym environments can be found under:

- ``robo_gym/__init__.py``

We have designed the following gym environments for the panda robot in simulation environment:
- ``'EmptyEnvironmentPandaSim-v0'``
- ``'EndEffectorPositioningPandaSim-v0'``
- ``'FollowPandaSim-v0'``

#### Testing:
For Evaluating the trained algorithm on the agent, you can run the code in the following:

```sh
python docs/examples/stable_baselines/EVAL/td3_script_with_eval.py

```

<!-- omit in toc -->
### [D4PG](https://github.com/schatty/oprl/tree/legacy_d4pg) algorithm

#### Training and Testing:

Open a new terminal and execute the following:

```sh
cd robo-gym-panda/docs/examples/d4pg-pytorch/
python3 train.py --config ./configs/openai/d4pg/panda_reach_d4pg.yml
```
- For training from beginning, training from a checkpoint and testing, you have to uncomment the respective lines in ``train.py``.
- All of the hyperparameters can be tuned under: 
  - ``robo-gym-panda/docs/examples/d4pg_pytorch/configs/openai/d4pg/panda_reach_d4pg.yml``

- Also the switch between ``'EmptyEnvironmentPandaSim-v0'``, ``'EndEffectorPositioningPandaSim-v0'`` can be done in this yaml file.

- When testing, or training from the checkpoint, the directory of the respective files has to be included in the yaml file.

- The ip of exploration and exploitation agents are to be set in:

  -``docs/examples/d4pg_pytorch/env/panda_wrapper.py`` (for the end-effector positioning environment)
  -``docs/examples/d4pg_pytorch/env/panda_follow_wrapper.py`` (for reach and follow environment)

- We suggest running the exploration and exploitation agents on seperate PCs to avoid overload and crashes. Also the GUI is only needed to be enabled for the exploitation agent for monitoring the training.


<!-- omit in toc -->
## Real Robot Environments

**NOTE:** Make sure to set up the target machine correctly when making the gym environment for real robot. For our case it was:

```sh
robot_address = '192.168.1.100:50051'
self.env = gym.make('EndEffectorPositioningPandaRob-v0', rs_address=robot_address)
```
This modification has to be done in ``docs/examples/d4pg_pytorch/env/panda_wrapper.py`` for the exploitation agent.

### Steps for testing on the real Franka Emika Panda robot:

First unlock joints and activate FCI

**Terminal 1:**

```sh
cd robogym_ws
./franka.sh master
roslaunch franka_interface interface.launch 
```
**Terminal 2:**
```sh
roslaunch panda_robot_server real_panda_robot_server.launch real_robot:=true gui:=true reference_frame:=world max_velocity_scale_factor:=0.1 action_cycle_rate:=400 objects_controller:=true rs_mode:=1object n_objects:=1.0 object_0_frame:=target action_mode:=abs_pos
```
**Terminal 3 (When testing D4PG):**
```sh
cd
cd robo-gym-panda/docs/examples/d4pg-pytorch
python3 train.py --config ./configs/openai/d4pg/panda_reach_d4pg.yml 
```
**NOTE:** No server manager needs to be started when testing (or training) on the real robot. The robot server has to be enabled manually as done above.

**NOTE:** Rviz can run through a 4th terminal for visualizatin purposes.

## General Notes and Issues

- EndEffectorPositioningPanda and ReachAndFollowPanda both inherit from the PandaBaseEnv class.
- For setting the maximum velocity, and switching between controllers:
	1. first you should modify the respective lines in ``PandaBaseEnv``
	2. In the ``FollowPandaSim`` or ``EndEffectorPositioningPandaSim`` where launch file is being called you should change the ``action_mode``. (``abs_pos`` for velocity mode and ``delta_pos`` for position difference controller )

- For debugging errors in the robot server side, open a new terminal:
  ```sh
		tmux -L ServerManager
  ```
  then: ``Crtl+B  )``
- Publishing frequency can be modified in ``FollowPandaSim`` or ``EndEffectorPositioningPandaSim`` at the bottom of the python files of the respective environments under the name ``action_cycle_rate``.

- Don't forget to add the following line to ``.bashrch``:
  ```sh  
		source /home/farhang/robogym_ws/devel/setup.bash 
  ```

----------------------------------------
