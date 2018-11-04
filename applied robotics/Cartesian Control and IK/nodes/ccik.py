#!/usr/bin/env python  
import math
import numpy
import rospy
import tf
import tf2_ros
import random
import geometry_msgs.msg
from sensor_msgs.msg import JointState 
from cartesian_control.msg import CartesianCommand
from urdf_parser_py.urdf import URDF
from geometry_msgs.msg import Transform
#####################################################################################################################
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

def SkewSymmetric(vector):
	S = numpy.zeros((3,3))

    	S[0,1] = -vector[2]
    	S[0,2] =  vector[1]
    	S[1,0] =  vector[2]
    	S[1,2] = -vector[0]
    	S[2,0] = -vector[1]
    	S[2,1] =  vector[0]
    	return S

def ComputeVelocity(b_T_ee_des, b_T_ee_cur):
		ee_cur_T_b = tf.transformations.inverse_matrix(b_T_ee_cur)
    		ee_cur_T_ee_des = numpy.dot(ee_cur_T_b, b_T_ee_des)

		Delta_X_tran = tf.transformations.translation_from_matrix(ee_cur_T_ee_des)
		
		angle,axis = rotation_from_matrix(ee_cur_T_ee_des)
		Delta_X_rot = numpy.dot(angle,axis)

		Delta_X = numpy.append(Delta_X_tran, Delta_X_rot, axis = 0)
		
		prop_gain = 1.4
		v_ee = Delta_X * prop_gain
		
		if numpy.linalg.norm(v_ee[0:3]) > 0.1:
			v_ee[0:3] =  v_ee[0:3] / (10 * numpy.linalg.norm(v_ee[0:3]))
		if numpy.linalg.norm(v_ee[3:6]) > 1:
			v_ee[3:6] = v_ee[3:6] / numpy.linalg.norm(v_ee[3:6])
		
		return (v_ee, Delta_X)

#####################################################################################################################
class CartesianControl:
	def __init__(self):
	# Loads the robot model
		self.robot = URDF.from_parameter_server()
	# Initialization
		self.num_joints = 0
		self.joints = []
		self.joint_names = []		
		self.all_axes = []
		self.current_joint_state = JointState()
        
	# Prepare general information about the robot
		self.get_joint_info()
	
	# Subscribes to information
		rospy.Subscriber("/joint_states", JointState, self.callback_jointstate) 
		rospy.sleep(.3) 
		rospy.Subscriber("/cartesian_command", CartesianCommand, self.callback_cartesiancommand)
		rospy.Subscriber("/ik_command", Transform, self.callback_inversekinematics)
		
	
	def callback_jointstate(self, joint_state):
		self.current_joint_state = joint_state
    
	# Cartesion Control
	def callback_cartesiancommand(self, user_command):
		sec_obj = user_command.secondary_objective
		q0_target = user_command.q0_target
		
		translation = tf.transformations.translation_matrix((user_command.x_target.translation.x,
                                                  user_command.x_target.translation.y,
                                                  user_command.x_target.translation.z))
    		rotation = tf.transformations.quaternion_matrix((user_command.x_target.rotation.x,
                                                user_command.x_target.rotation.y,
                                                user_command.x_target.rotation.z,
                                                user_command.x_target.rotation.w))
    		x_desired = numpy.dot(translation,rotation)
		
		all_transforms, x_current = self.forward_kinematics()
		
		v_ee, Delta_X = ComputeVelocity(x_desired, x_current)
		
		J = self.AssembleJacobian(all_transforms, x_current)

		J_pinv_safe = numpy.linalg.pinv(J, 1.0e-2)
		qDot = numpy.dot(J_pinv_safe, v_ee)
		
		if sec_obj == True:

			qNull = self.SecondaryObject(J, q0_target)
			qDot = qDot + qNull

		if max(qDot) > 1:
			qDot /= max(qDot)
	
		self.Publish_qDot(qDot)
	# Numerical IK
	def callback_inversekinematics(self, transform):
		
		translation = tf.transformations.translation_matrix((transform.translation.x,
                                                  transform.translation.y,
                                                  transform.translation.z))
    		rotation = tf.transformations.quaternion_matrix((transform.rotation.x,
                                                transform.rotation.y,
                                                transform.rotation.z,
                                                transform.rotation.w))
    		x_desired = numpy.dot(translation,rotation)
		# The total iteration is 3
		for k in range(3):
			# 
			q_current = [0] * self.num_joints
			for j in range(self.num_joints):
				q_current[j] = random.uniform(0, math.pi)
			# Represent the time interval with the number of iteration 
			for i in range(300):

				all_transforms, x_current = self.forward_kinematics_2(q_current)
				v_ee, Delta_X = ComputeVelocity(x_desired, x_current)
			
				if max(abs(Delta_X)) < 0.01:
					break

				J = self.AssembleJacobian(all_transforms, x_current)
				J_pinv_safe = numpy.linalg.pinv(J, 0.01)
				Delta_q = numpy.dot(J_pinv_safe, Delta_X)
				q_current = q_current + Delta_q
			if max(abs(Delta_X)) < 0.01:
					break
		
		self.Publish_JointState(q_current)

#######   self-defined functions within class
	def get_joint_info(self):
		self.num_joints = 0
		self.joints = []
		self.joint_names = []
		self.all_axes = []
		
		link_name = self.robot.get_root()
		while True:
			if link_name not in self.robot.child_map: 
				break

			(joint_name, next_link_name) = self.robot.child_map[link_name][0]

			current_joint = self.robot.joint_map[joint_name]
			self.joints.append(current_joint)

			if current_joint.type != 'fixed':
				self.num_joints = self.num_joints + 1
				self.all_axes.append(current_joint.axis)
				self.joint_names.append(current_joint.name)
				
			link_name = next_link_name

	def forward_kinematics(self):
		all_transforms = []
		index_fixed = []		
		
		T = tf.transformations.identity_matrix()
		for i in range(len(self.joints)):
		
			T1 = tf.transformations.translation_matrix(self.joints[i].origin.xyz)
			r = self.joints[i].origin.rpy[0]
			p = self.joints[i].origin.rpy[1]
			y = self.joints[i].origin.rpy[2]
			T2 = tf.transformations.euler_matrix(r,p,y)
			
			if self.joints[i].type == "revolute" :
				j = self.current_joint_state.name.index(self.joints[i].name)
				q = self.current_joint_state.position[j]
				
				T3 =tf.transformations.rotation_matrix(q, self.joints[i].axis)			
			else:
				T3 = tf.transformations.identity_matrix()
				index_fixed.append(i)

			T = tf.transformations.concatenate_matrices(T,T1,T2,T3)
			all_transforms.append(T)

		b_T_ee_cur = all_transforms[-1]
		
		for j in sorted(index_fixed, reverse=True):
			del all_transforms[j]

		return (all_transforms, b_T_ee_cur)

	def forward_kinematics_2(self, joint_state):
		all_transforms = []
		index_fixed = []		
		
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

		b_T_ee_cur = all_transforms[-1]
		
		for j in sorted(index_fixed, reverse=True):
			del all_transforms[j]

		return (all_transforms, b_T_ee_cur)

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

	def SecondaryObject(self, J, q0_target):
		
		qSec = [0] * self.num_joints

		qSec[0] = (q0_target - self.current_joint_state.position[0]) * 1.4
		J_pinv = numpy.linalg.pinv(J)
		I = numpy.identity(self.num_joints)
		N = I - numpy.dot(J_pinv, J)
		qNull = numpy.dot(N, qSec)

		return qNull		
#### Publish qDot in cartesian control
	def Publish_qDot(self, qDot):
		pub_vel = rospy.Publisher("/joint_velocities", JointState, queue_size=1)
		msg = JointState()
		

		msg.velocity = qDot
		msg.name = self.joint_names

		pub_vel.publish(msg)

### Publish jointstate in Numerical IK
	def Publish_JointState(self, jointstate):
		pub_js = rospy.Publisher("/joint_command", JointState, queue_size=1)
		msg = JointState()

		msg.position = jointstate
		msg.name = self.joint_names

		pub_js.publish(msg)


if __name__ == '__main__':
   	rospy.init_node('cartesian_controller', anonymous=True)
   	cc = CartesianControl()
   	rospy.spin()


	
