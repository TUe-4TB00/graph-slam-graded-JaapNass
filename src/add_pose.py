
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # TODO: Add the odometry factor between X(3) and X(4) to the graph (BetweenFactorPose2)
    delta_x     = np.sqrt(2.0)
    delta_y     = np.sqrt(2.0)
    delta_theta = np.pi/2.0

    odometry    = gtsam.Pose2(delta_x, delta_y, delta_theta)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    # TODO: Based on the odometry, find the initial estimate for the pose of X(4) and add it to the graph
    pose3_guess     = initial_estimate.atPose2(X(3))
    pose4_guess     = pose3_guess.compose(odometry)
    initial_estimate.insert(X(4), pose4_guess)

    params          = gtsam.LevenbergMarquardtParams()
    optimizer       = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result          = optimizer.optimize()

    pose3_optimized = result.atPose2(X(3))
    pose4_optimized = pose3_optimized.compose(odometry)

    initial_estimate.update(X(4), pose4_optimized)


    return graph, initial_estimate