# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from aerial_gym.envs.base.aerial_robot_config import AerialRobotCfg
from .base.dynamics_trainer import DynamicsTrainer
from .base.track_ground import TrackGround
from .base.track_ground_test import TrackGroundTest
from .base.track_simple import TrackSimple
from .base.track_simple_config import TrackSimpleCfg
from .base.track_ground_config import TrackGroundCfg
from .base.dynamics_learnt import LearntDynamics
from .base.dynamics_isaac import IsaacGymDynamics
from .base.track_groundVer2 import TrackGroundVer2
from .base.track_groundVer3 import TrackGroundVer3
from .base.track_groundVer4 import TrackGroundVer4
from .base.track_groundVer5 import TrackGroundVer5
from .base.track_groundVer6 import TrackGroundVer6
from .base.track_groundVer7 import TrackGroundVer7
from .base.track_groundVer8 import TrackGroundVer8
from .base.track_groundVer9 import TrackGroundVer9
from .base.track_groundVer10 import TrackGroundVer10
from aerial_gym.utils.task_registry import task_registry

# task_registry.register( "quad", AerialRobot, AerialRobotCfg())
# task_registry.register("quad_with_obstacles", AerialRobotWithObstacles, AerialRobotWithObstaclesCfg())
task_registry.register( "train_dynamics", DynamicsTrainer, AerialRobotCfg())
task_registry.register( "track_simple", TrackSimple, TrackSimpleCfg())
task_registry.register( "track_ground", TrackGround, TrackGroundCfg())
task_registry.register( "track_ground_test", TrackGroundTest, TrackGroundCfg())
task_registry.register( "track_groundVer2", TrackGroundVer2, TrackGroundCfg())
task_registry.register( "track_groundVer3", TrackGroundVer3, TrackGroundCfg())
task_registry.register( "track_groundVer4", TrackGroundVer4, TrackGroundCfg())
task_registry.register( "track_groundVer5", TrackGroundVer5, TrackGroundCfg())
task_registry.register( "track_groundVer6", TrackGroundVer6, TrackGroundCfg())
task_registry.register( "track_groundVer7", TrackGroundVer7, TrackGroundCfg())
task_registry.register( "track_groundVer8", TrackGroundVer8, TrackGroundCfg())
task_registry.register( "track_groundVer9", TrackGroundVer9, TrackGroundCfg())
task_registry.register( "track_groundVer10", TrackGroundVer10, TrackGroundCfg())