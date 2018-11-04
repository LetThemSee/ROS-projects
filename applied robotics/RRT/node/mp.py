#!/usr/bin/env python
import numpy
import sys

import trajectory_msgs
import sensor_msgs.msg
import geometry_msgs.msg
import moveit_msgs.msg
import moveit_msgs.srv
import rospy
import tf

import random
import math
import time
from urdf_parser_py.urdf import URDF

class Node(object):
	
		def __init__(self, joint_state, parent = None):
			self.joint_state = joint_state
			self.parent = parent
		def get_js(self):
			return self.joint_state
		def get_parent(self):
			return self.parent

def compute_dist(q1, q2):
		diff = numpy.subtract(q2, q1)
		dist = numpy.linalg.norm(diff)
		
		return dist

def compute_unit_vector(q_start, q_goal):
		diff = numpy.subtract(q_goal, q_start)
		unit_vector = diff / numpy.linalg.norm(diff)
		
		return unit_vector

def convert_to_message(T):
		t = geometry_msgs.msg.Pose()
		position = tf.transformations.translation_from_matrix(T)
		orientation = tf.transformations.quaternion_from_matrix(T)
		t.position.x = position[0]
		t.position.y = position[1]
		t.position.z = position[2]
		t.orientation.x = orientation[0]
		t.orientation.y = orientation[1]
		t.orientation.z = orientation[2]
		t.orientation.w = orientation[3]        
		return t

#################################################################
## auxiliary functions to implement IK of my version
def rotation_from_matrix(matrix):
	R = numpy.array(matrix, dtype=numpy.float64, copy=False)
	R33 = R[:3, :3]
	# axis: unit eigenvector of R33 corresponding to eigenvalue of 1
	l, W = numpy.linalg.eig(R33.T)
	i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
	if not len(i):
		raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
	axis = numpy.real(W[:, i[-1]]).squeeze()
	# point: unit eigenvector of R33 corresponding to eigenvalue of 1
	l, Q = numpy.linalg.eig(R)
	i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
	if not len(i):
		raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
	# rotation angle depending on axis
	cosa = (numpy.trace(R33) - 1.0) / 2.0
	if abs(axis[2]) > 1e-8:
		sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
	elif abs(axis[1]) > 1e-8:
		sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
	else:
		sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
	angle = math.atan2(sina, cosa)

	return angle, axis

def compute_deltaX(b_T_ee_des, b_T_ee_cur):
		ee_cur_T_b = tf.transformations.inverse_matrix(b_T_ee_cur)
    		ee_cur_T_ee_des = numpy.dot(ee_cur_T_b, b_T_ee_des)

		Delta_X_tran = tf.transformations.translation_from_matrix(ee_cur_T_ee_des)
		
		angle,axis = rotation_from_matrix(ee_cur_T_ee_des)
		Delta_X_rot = numpy.dot(angle,axis)

		Delta_X = numpy.append(Delta_X_tran, Delta_X_rot, axis = 0)
		
		return Delta_X

def SkewSymmetric(vector):
	S = numpy.zeros((3,3))

    	S[0,1] = -vector[2]
    	S[0,2] =  vector[1]
    	S[1,0] =  vector[2]
    	S[1,2] = -vector[0]
    	S[2,0] = -vector[1]
    	S[2,1] =  vector[0]

    	return S

#######################################################################################################
class MoveArm(object):

	def __init__(self):

		# Wait for moveit IK service
		rospy.wait_for_service("compute_ik")
		self.ik_service = rospy.ServiceProxy('compute_ik',  moveit_msgs.srv.GetPositionIK)
		print "IK service ready"

		# Wait for validity check service
		rospy.wait_for_service("check_state_validity")
		self.state_valid_service = rospy.ServiceProxy('check_state_validity',  
			                                      moveit_msgs.srv.GetStateValidity)
		print "State validity service ready"

		# MoveIt parameter
		self.group_name = "lwr_arm"
		

		# Obtain basic information to implement my IK	
		self.robot = URDF.from_parameter_server()

		self.num_joints = 0
		self.joints = []
		self.joint_names = []		
		self.all_axes = []
		self.current_joint_state = sensor_msgs.msg.JointState()

		self.get_joint_info()
		
		# Initialize the publisher
		self.pub_trajectory = rospy.Publisher("/joint_trajectory", trajectory_msgs.msg.JointTrajectory, queue_size=1)
		# Subscribe to topics
		rospy.Subscriber("/joint_states", sensor_msgs.msg.JointState, self.callback_jointstate) 
		rospy.Subscriber("/motion_planning_goal", geometry_msgs.msg.Transform, self.callback_get_goal) 

    	def callback_jointstate(self, joint_state):
		self.current_joint_state = joint_state

	def callback_get_goal(self, des_goal):
		translation = tf.transformations.translation_matrix((des_goal.translation.x,
                                                  des_goal.translation.y,
                                                  des_goal.translation.z))
    		rotation = tf.transformations.quaternion_matrix((des_goal.rotation.x,
                                                des_goal.rotation.y,
                                                des_goal.rotation.z,
                                                des_goal.rotation.w))
    		goal_Transform = numpy.dot(translation,rotation)

#######################################################################################################		
		goal_joint_state = self.IK(goal_Transform)
		#goal_joint_state = self.my_IK(goal_Transform)
		print self.current_joint_state.name
#######################################################################################################

		start_joint_state = self.current_joint_state.position

		# Avoid special cases
		if goal_joint_state == []:
			print ''
			print 'No such IK solution is Found'
			print 'Try other goal.'
			exit()
		if not self.is_state_valid(start_joint_state):
			print ''
			print 'The starting position is colliding with obstacle'
			print 'Remove the obstacle and try again.'
			exit()
		if not self.is_state_valid(goal_joint_state):
			print ''
			print 'The goal is colliding with obstacle'
			print 'Try other goal.'
			exit()

		#######################################################################################################
		# Implement RRT:
		print ""
		print "Looking for a path..."

		begin = rospy.get_time()
		print 'start time', begin

		tree = list()
		start = Node(start_joint_state)
		tree.append(start)
		step_size = 0.5
		
   		# While cannot connect to the goal, keep sampling and generate a tree
		while True:
			# Check if the newly added node in tree is able to connect to the goal
			if self.is_path_collision_free(tree[-1].get_js(), goal_joint_state):
				break

			# Samle random point R in configuration space
			q_sample = [random.uniform(-math.pi, math.pi) for i in range(len(goal_joint_state))]

			# Find the closest point P in tree to point R
			dist = 10000
			for i in range(len(tree)) :
				if compute_dist(tree[i].get_js(), q_sample) < dist:
					dist = compute_dist(tree[i].get_js(), q_sample)
					parent = tree[i]

			# Add branch of predefined length in the direction from P to R
			# if P to R is collision free
			unit_vector = compute_unit_vector(parent.get_js(), q_sample)
			next_joint_state = parent.get_js() + unit_vector * step_size

			if self.is_path_collision_free(parent.get_js(), next_joint_state):
				next_node = Node(next_joint_state, parent)
				tree.append(next_node)

		print 'tree (number of node)'
		print len(tree)
		if len(tree) > 1000:
			print "Time is up..."
			print "Please restart"
			exit()

		# Trace backward to obtain a trajectory
		trajectory = list()
		cur_node = tree[-1]
			# starting node's parent is None by default
		while cur_node.parent is not None:
			trajectory.append(cur_node.get_js())
			cur_node = cur_node.parent

		trajectory.append(start_joint_state)
		trajectory.reverse()
		trajectory.append(goal_joint_state)

		print "trajectory (number of nodes)"
		print len(trajectory)
		
		# Perform shortcut
		new_trajectory = []
		new_trajectory.append(trajectory[0])
		i = 0
		while i < len(trajectory):
			j = i + 1
			while j < len(trajectory):
				if not self.is_path_collision_free(trajectory[i], trajectory[j]):
					new_trajectory.append(trajectory[j-1])
					break
				j = j + 1

			if j == len(trajectory):
				new_trajectory.append(trajectory[-1])
				break
			else:
				i = j - 1

		print "after shortcut"
		print len(new_trajectory)

		# Add a number of internal samples equally spaced apart inside the segment. 
		final_trajectory = list()
		for i in range(len(new_trajectory) - 1):
			
			final_trajectory.append(new_trajectory[i])
			final_trajectory += self.space_apart_segment(new_trajectory[i], new_trajectory[i+1])
			final_trajectory.append(new_trajectory[i+1])
		
		# Publish final trajectory 
		self.publish_trajectory(final_trajectory)

		end = rospy.get_time()
		print 'end time', end
		print 'Time cost', end - begin

    # This function will perform IK for a given transform T of the end-effector. It 
    # returns a list q[] of 7 values, which are the result positions for the 7 joints of 
    # the KUKA arm, ordered from proximal to distal. If no IK solution is found, it 
    # returns an empty list.
 
    	def IK(self, T_goal):
		req = moveit_msgs.srv.GetPositionIKRequest()
		req.ik_request.group_name = self.group_name
		req.ik_request.robot_state = moveit_msgs.msg.RobotState()
		req.ik_request.robot_state.joint_state.name = ["lwr_arm_0_joint",
			                                       "lwr_arm_1_joint",
			                                       "lwr_arm_2_joint",
			                                       "lwr_arm_3_joint",
			                                       "lwr_arm_4_joint",
			                                       "lwr_arm_5_joint",
			                                       "lwr_arm_6_joint"]
		req.ik_request.robot_state.joint_state.position = numpy.zeros(7)
		req.ik_request.robot_state.joint_state.velocity = numpy.zeros(7)
		req.ik_request.robot_state.joint_state.effort = numpy.zeros(7)
		req.ik_request.robot_state.joint_state.header.stamp = rospy.get_rostime()
		req.ik_request.avoid_collisions = True
		req.ik_request.pose_stamped = geometry_msgs.msg.PoseStamped()
		req.ik_request.pose_stamped.header.frame_id = "world_link"
		req.ik_request.pose_stamped.header.stamp = rospy.get_rostime()
		req.ik_request.pose_stamped.pose = convert_to_message(T_goal)
		req.ik_request.timeout = rospy.Duration(3.0)
		res = self.ik_service(req)
		q = []
		if res.error_code.val == res.error_code.SUCCESS:
			q = res.solution.joint_state.position
		return q

    # This is my version of IK and its auxiliary functions 
	def my_IK(self, T_goal):
		
		q_current = [random.uniform(0, math.pi) for i in range(self.num_joints)]

		for j in range(3):

			for i in range(300):

				all_transforms, x_current = self.forward_kinematics(q_current)
				Delta_X = compute_deltaX(T_goal, x_current)

				if max(abs(Delta_X)) < 0.001:
					break

				J = self.AssembleJacobian(all_transforms, x_current)

				J_pinv = numpy.linalg.pinv(J)
				Delta_q = numpy.dot(J_pinv, Delta_X)
			
				q_current = q_current + Delta_q
			
			if max(abs(Delta_X)) < 0.001:
				break

		if not self.is_state_valid(q_current) :
			q_current = []
		if j == 3:
			q_current = []

		for i in range(len(q_current)):
			while q_current[i] <= -math.pi: 
				q_current[i] += 2 * math.pi
    			while q_current[i] > math.pi:
				q_current[i] -= 2 * math.pi
		
		return q_current

	def get_joint_info(self):
		self.num_joints = 0
		self.joints = []
		self.joint_names = []
		self.all_axes = []
		
		link_name = self.robot.get_root()
		while True:
			if link_name not in self.robot.child_map: 
				break

			joint_name, next_link_name = self.robot.child_map[link_name][0]

			current_joint = self.robot.joint_map[joint_name]
			self.joints.append(current_joint)

			if current_joint.type != 'fixed':
				self.num_joints = self.num_joints + 1
				self.all_axes.append(current_joint.axis)
				self.joint_names.append(current_joint.name)
				
			link_name = next_link_name
		
	def forward_kinematics(self, joint_state):
		all_transforms = list()
		index_fixed = list()		
		
		T = tf.transformations.identity_matrix()
		for i in range(len(self.joints)):
		
			T1 = tf.transformations.translation_matrix(self.joints[i].origin.xyz)
			r = self.joints[i].origin.rpy[0]
			p = self.joints[i].origin.rpy[1]
			y = self.joints[i].origin.rpy[2]
			T2 = tf.transformations.euler_matrix(r,p,y)
			
			if self.joints[i].type == "revolute" :
				j = self.joint_names.index(self.joints[i].name)
				q = joint_state[j]
				T3 =tf.transformations.rotation_matrix(q, self.joints[i].axis)			
			else:
				T3 = tf.transformations.identity_matrix()
				index_fixed.append(i)

			T = tf.transformations.concatenate_matrices(T,T1,T2,T3)
			all_transforms.append(T)

		x_current = all_transforms[-1]
		
		for j in sorted(index_fixed, reverse=True):
			del all_transforms[j]

		return all_transforms, x_current

	def AssembleJacobian(self, all_transforms, b_T_ee_cur):
		Jacobian = numpy.empty((6, 0))

		for j in range(self.num_joints):

			b_T_j = all_transforms[j]
			j_T_b = tf.transformations.inverse_matrix(b_T_j)
			j_T_ee = numpy.dot(j_T_b, b_T_ee_cur)
			
			j_tran_ee = tf.transformations.translation_from_matrix(j_T_ee) 
			S = SkewSymmetric(j_tran_ee)

			ee_T_j = tf.transformations.inverse_matrix(j_T_ee)
			ee_rot_j = ee_T_j[:3,:3]

			V_j_up = numpy.append(ee_rot_j, numpy.dot(-ee_rot_j, S), axis=1) 
			V_j_down = numpy.append(numpy.zeros([3,3]), ee_rot_j, axis=1)
			V_j = numpy.append(V_j_up, V_j_down, axis=0)
			
			axis = self.all_axes[j]
			for index in range(len(axis)):
				if axis[index] == 1:
					Jacobian = numpy.column_stack((Jacobian, V_j[:,index+3])) 
					break
				if axis[index] == -1:
					Jacobian = numpy.column_stack((Jacobian, -V_j[:,index+3])) 
					break			

		return Jacobian
	# This function checks if a set of joint angles q[] creates a valid state, or 
	# one that is free of collisions. The values in q[] are assumed to be values for 
	# the joints of the KUKA arm, ordered from proximal to distal. 

    	def is_state_valid(self, q):
		req = moveit_msgs.srv.GetStateValidityRequest()
		req.group_name = self.group_name
		req.robot_state = moveit_msgs.msg.RobotState()
		req.robot_state.joint_state.name = ["lwr_arm_0_joint",
			                            "lwr_arm_1_joint",
			                            "lwr_arm_2_joint",
			                            "lwr_arm_3_joint",
			                            "lwr_arm_4_joint",
			                            "lwr_arm_5_joint",
			                            "lwr_arm_6_joint"]
		req.robot_state.joint_state.position = q
		req.robot_state.joint_state.velocity = numpy.zeros(7)
		req.robot_state.joint_state.effort = numpy.zeros(7)
		req.robot_state.joint_state.header.stamp = rospy.get_rostime()
		res = self.state_valid_service(req)
		return res.valid

	########################################################################################################################
	# Self defined functions
	# Space apart a segment equally and add each part into a path, but without start and goal
	def space_apart_segment(self, segment_start, segment_goal):

		dist = compute_dist(segment_start, segment_goal)
		unit_vector = compute_unit_vector(segment_start, segment_goal)
		num_interval = int(math.floor(dist / 0.1))
		
		path = list()
		current = segment_start
		for i in range(num_interval):
			current = current + unit_vector * 0.1
			path.append(current)

		return path
	# Check if a path is collision-free
	def is_path_collision_free(self, segment_start, segment_goal):
		if not self.is_state_valid(segment_goal):
			return False

		dist = compute_dist(segment_start, segment_goal)
		unit_vector = compute_unit_vector(segment_start, segment_goal)
		num_interval = int(math.floor(dist / 0.1))
		
		current = segment_start
		for i in range(num_interval):
			current = current + unit_vector * 0.1
			if not self.is_state_valid(current):
				return False

		return True

	def publish_trajectory(self, trajectory):
		#pub_trajectory = rospy.Publisher("/joint_trajectory", trajectory_msgs.msg.JointTrajectory, queue_size=1)
		msg = trajectory_msgs.msg.JointTrajectory()
		
		for i in range(len(trajectory)):
			point = trajectory_msgs.msg.JointTrajectoryPoint()
			point.positions = list(trajectory[i])
			msg.points.append(point)

		msg.joint_names = self.current_joint_state.name
		
		# pub_trajectory.publish(msg)
		self.pub_trajectory.publish(msg)
	

if __name__ == '__main__':
	rospy.init_node('move_arm', anonymous=True)
	ma = MoveArm()
	rospy.spin()

