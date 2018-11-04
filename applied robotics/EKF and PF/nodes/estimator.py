#!/usr/bin/env python

import rospy
import math
import numpy

from state_estimator.msg import SensorData
from geometry_msgs.msg import Pose2D

import random

class Particle(object):
	def __init__(self, x = None, weight = None):
		self.x = x
		self.weight = weight

	def get_state(self):
		return self.x

	def get_weight(self):
		return self.weight
	
	def set_weight(self, new_weight):
		self.weight = new_weight

class Estimator(object):

    	def __init__(self):	

		# Initialize parameters
		self.t = 0.01
		
		self.cur_sensor_data = SensorData()

		self.vel_trans = 0
		self.vel_ang = 0
		self.X_l = []
		self.Y_l = []
		self.Range_from_sensor = []
		self.Bearing_from_sensor = []
		
		# Prepare publisher
		self.pub_est_state = rospy.Publisher("/robot_pose_estimate", Pose2D, queue_size = 1)
		# Subscribe to get sensor data
		rospy.Subscriber("/sensor_data", SensorData, self.callback_sensor_data)

		####################################################################################
		### Extended Kalmon Filter initialization

                # Start point assumed by estimator
		self.x_est = numpy.array([[0], [0], [0]])
		self.P_est = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

		# White noise
		self.V = numpy.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
		self.W = numpy.array([[0.1, 0], [0, 0.1]])
		
		####################################################################################
		### Particle filter initialization
		
		self.all_particle_sets = []
		self.num_particles = 100

		# Initialize a particle set based on our initial knowledge and add it to all_particle_sets
		self.particle_set = [ Particle(numpy.array([[0], [0], [0]]), 1/self.num_particles) for i in range(self.num_particles) ]
		self.all_particle_sets.append(self.particle_set)

	def callback_sensor_data(self, sensor_data):
		self.cur_sensor_data = sensor_data
		#print self.cur_sensor_data
		self.ext_Kalman_step()

	def get_info(self):
		self.X_l = []
		self.Y_l = []
		self.Range_from_sensor = []
		self.Bearing_from_sensor = []

		for i in range(len(self.cur_sensor_data.readings)):
			self.X_l.append(self.cur_sensor_data.readings[i].landmark.x)
			self.Y_l.append(self.cur_sensor_data.readings[i].landmark.y)

			self.Range_from_sensor.append(self.cur_sensor_data.readings[i].range)
			self.Bearing_from_sensor.append(self.cur_sensor_data.readings[i].bearing)
		
		self.vel_trans = self.cur_sensor_data.vel_trans
		self.vel_ang = self.cur_sensor_data.vel_ang
	
	def ext_Kalman_step(self):
		# Get information 
		self.get_info()
		
		# Wait for initializing the robot
 		if self.vel_trans != 0 or self.vel_ang != 0:
	
			# Advance one step
			### 1.Linearize F based on x_estimate(current)
			F_linearized = self.linearize_F()

			### 2.Prediction step
			x_pred = self.calculate_next_step(self.x_est) 
			P_pred = numpy.dot(F_linearized, numpy.dot(self.P_est, numpy.transpose(F_linearized))) + self.V

			### Delete sensor data that could possibly cause singular matrix error
			#to_be_del = []
			#for i in range(len(self.X_l)):
				#if self.is_remove_sensor_data(x_pred, self.X_l[i], self.Y_l[i]):
					#to_be_del.append(i)
			#for j in sorted(to_be_del, reverse=True):
				#del self.X_l[j]
				#del self.Y_l[j]
				#del self.Range_from_sensor[j]
				#del self.Bearing_from_sensor[j]
			
			### 3.Calculate necessary parameters for the updating step
			num_landmark = len(self.X_l)
			if num_landmark != 0:
				
				H = numpy.empty((0,3))
				innovation = numpy.empty((0, 1))
				R = numpy.empty((3, 0))
				flag = 0

				for i in range(num_landmark):
					### Linearize H based on x_prediction and 
					H_linearized = self.linearize_H(x_pred, self.X_l[i], self.Y_l[i])
					H = numpy.row_stack((H, H_linearized))

					### Calculate range and bearing without noise
					sensor_without_noise = self.calculate_expected_range_and_bearing(x_pred, self.X_l[i], self.Y_l[i])

					### Calculate Innovation / S / R
					innov = numpy.array([[self.Range_from_sensor[i]], [self.Bearing_from_sensor[i]]]) - sensor_without_noise
					innovation = numpy.row_stack((innovation, innov))
						
					S = numpy.dot( H_linearized, numpy.dot(P_pred, numpy.transpose(H_linearized)) ) + self.W
					#try:
						#inv_S = numpy.linalg.inv(S)
					#except:
						#flag = 1
						#break
					R_cur = numpy.dot(P_pred, numpy.dot(numpy.transpose(H_linearized), numpy.linalg.pinv(S, 0.001) ) )
					R = numpy.column_stack((R, R_cur))

				### 4.Update step
				if flag:
					delta_x = numpy.array([[0], [0], [0]])
					delta_P = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
				else:
					delta_x = numpy.dot(R, innovation)
					delta_P = -1.0 * numpy.dot( R, numpy.dot(H, P_pred) )	

			else:
				### 4.Update step
				delta_x = numpy.array([[0], [0], [0]])
				delta_P = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

			self.x_est = x_pred + delta_x
			self.P_est = P_pred + delta_P

	
		# Publish new estimated state
		self.publish_estimate_state(self.x_est)

	def particle_filter(self): 
		# Get information 
		self.get_info()
			
		# Wait until the robot is initialized
 		if self.vel_trans != 0 or self.vel_ang != 0:
		
			temp_particle_set = []
			sum_weight = 0
			num_landmark = len(self.X_l)

			for particle_est in self.particle_set:
				# Step 1: Calculate next step for each particle
				x_est = particle_est.get_state()

				# Method 1: Consider noise after predicting
				#white_noise = self.generate_white_noise()
				#x_pred = self.calculate_next_step(x_est) + white_noise

				# Method 2: Take the noise into consideration before predicting
		  		x_pred = self.calculate_next_step_PF(x_est)

				# Step 2: Calculate weight(probability) for each particle based on the difference 
				#	  between expected and received sensor data
				
				if num_landmark != 0:

					difference = numpy.empty((0, 1))
					for i in range(num_landmark):
						exp_sensor = self.calculate_expected_range_and_bearing(x_pred, self.X_l[i], self.Y_l[i])
						diff = numpy.array([[self.Range_from_sensor[i]], [self.Bearing_from_sensor[i]]]) - exp_sensor
						difference = numpy.row_stack((difference, diff))

					weight = 1 / numpy.linalg.norm(difference)

				else:
					
					weight = 1 / self.num_particles

				sum_weight += weight
				temp_particle_set.append(Particle(x_pred, weight))
				
			# Step 3: Generate a new particle set
			new_particle_set = []
		
			if num_landmark != 0:
				# Step 4: Select particles from temporary particle set(X_prediction set) based on their weights
				# Step 5: Repeat until N particles have been in new particle set(X_estimate set)
				for i in range(self.num_particles):
					lottery = random.uniform(0, sum_weight)
					pool = sum_weight
					for particle_pred in temp_particle_set:
						weight = particle_pred.get_weight()
						pool = pool - weight
						if pool < lottery:
							new_particle_set.append(particle_pred)
							break
						
			else:
				new_particle_set = temp_particle_set[:]

			self.particle_set = new_particle_set[:]
			self.all_particle_sets.append(self.particle_set)
						
		# Find the particle with highest weight to represent optimal state
		optimal_state = self.find_optimal_state()
		
		# Publish new estimated state
		self.publish_estimate_state(optimal_state)
	
####################################################################################################################################
### Self-defined functions
	def linearize_F(self):
		theta = float(self.x_est[2])
		F_linearized = numpy.array([ [1, 0, -self.t * self.vel_trans * math.sin(theta)],
				             [0, 1, self.t * self.vel_trans * math.cos(theta)],
				             [0, 0, 1] ])

		return F_linearized

	def linearize_H(self, x_pred, x_l, y_l):
		x_r = float(x_pred[0])
		y_r = float(x_pred[1])
		theta_r = float(x_pred[2])

		denominator = pow(x_r - x_l, 2) + pow(y_r - y_l, 2)

		dh1_dx = (x_r - x_l) / math.sqrt (denominator)
		dh1_dy = (y_r - y_l) / math.sqrt (denominator)

		dh2_dx = (y_l - y_r) / denominator
		dh2_dy = (x_r - x_l) / denominator

		H_linearized = numpy.array([ [dh1_dx, dh1_dy, 0],
				     	     [dh2_dx ,dh2_dy, -1] ])
		
		return H_linearized

	def calculate_next_step(self, x_est):
		x = float(x_est[0])
		y = float(x_est[1])
		theta = float(x_est[2])

		x_pred = numpy.array([ [x + self.t * self.vel_trans * math.cos(theta)], 
				       [y + self.t * self.vel_trans * math.sin(theta)], 
				       [theta + self.t * self.vel_ang] ])
		return x_pred
	
	def calculate_expected_range_and_bearing(self, x_pred, x_l, y_l):
		x_r = float(x_pred[0])
		y_r = float(x_pred[1])
		theta_r = float(x_pred[2])

		range_no_noise = math.sqrt( (x_r - x_l) * (x_r - x_l) + (y_r - y_l) * (y_r - y_l) )
		bearing_no_noise = math.atan2(y_l-y_r, x_l-x_r) - theta_r
		
		return numpy.array([[range_no_noise], [bearing_no_noise]])
	
	def is_remove_sensor_data(self, x_pred, x_l, y_l):
		x_r = float(x_pred[0])
		y_r = float(x_pred[1])
		theta_r = float(x_pred[2])
		
		if abs(x_r - x_l) < 0.001 or abs(y_r - y_l) < 0.001 :
			return True
		else:
			return False

	def publish_estimate_state(self, x_est): 
	
		msg = Pose2D()
		msg.x = x_est[0]
		msg.y = x_est[1]
		msg.theta = x_est[2]

		self.pub_est_state.publish(msg)

####################################################################################################################################
### Additional functions for particle filter
	def generate_white_noise(self):
		mu = 0
		sigma = math.sqrt(0.1)
		white_noise = numpy.array([[numpy.random.normal(0.0, 0.1)], [numpy.random.normal(0.0, 0.1)], [numpy.random.normal(0.0, 0.1)]])
		
		return white_noise

	def calculate_next_step_PF(self, x_est):

		x = float(x_est[0]) + numpy.random.normal(0.0, 0.1)
		y = float(x_est[1]) + numpy.random.normal(0.0, 0.1)
		theta = float(x_est[2]) + numpy.random.normal(0.0, 0.1)
		vel_trans = self.vel_trans + numpy.random.normal(0.0, 0.1)
		vel_ang = self.vel_ang + numpy.random.normal(0.0, 0.1)

		x_pred = numpy.array([ [x + self.t * vel_trans * math.cos(theta)], 
				       [y + self.t * vel_trans * math.sin(theta)], 
				       [theta + self.t * vel_ang] ])
		return x_pred

	def find_optimal_state(self):
		highest_weight = -1
		optimal_state = []
		for particle in self.particle_set:
			weight = particle.get_weight()
			x_est = particle.get_state()
			if weight > highest_weight:
				highest_weight = weight
				optimal_state = x_est

		return optimal_state

if __name__ == '__main__':
	rospy.init_node('estimate_state', anonymous=True)
	estimator = Estimator()
	rospy.spin()

	#while not rospy.is_shutdown():
		
		#estimator.ext_Kalman_step()
		#estimator.particle_filter()
		
		#rospy.sleep(0.01)
        	
