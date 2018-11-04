#!/usr/bin/env python  
import math
import numpy
import rospy
import tf
import geometry_msgs.msg
from sensor_msgs.msg import JointState 
from urdf_parser_py.urdf import URDF
import tf.msg
import tf2_ros



def message_from_transform(T,parent,child):
	msg = geometry_msgs.msg.TransformStamped()
	
	msg.header.stamp = rospy.Time.now()
    	msg.header.frame_id = parent
  	msg.child_frame_id = child
	
    	translation = tf.transformations.translation_from_matrix(T)
    	rotation = tf.transformations.quaternion_from_matrix(T)

    	msg.transform.translation.x = translation[0]
    	msg.transform.translation.y = translation[1]
    	msg.transform.translation.z = translation[2]
   	msg.transform.rotation.x = rotation[0]
   	msg.transform.rotation.y = rotation[1]
    	msg.transform.rotation.z = rotation[2]
    	msg.transform.rotation.w = rotation[3] 
	
	return msg

def calculate_and_publish_transforms(link_names, joints, joint_values):
	br = tf2_ros.TransformBroadcaster()
	T = tf.transformations.identity_matrix()

	for i in range(len(joints)) :
		
		T1 = tf.transformations.translation_matrix(joints[i].origin.xyz)
		r = joints[i].origin.rpy[0]
		p = joints[i].origin.rpy[1]
		y = joints[i].origin.rpy[2]
		T2 = tf.transformations.euler_matrix(r,p,y)

		if joints[i].type == "revolute" :
			j = joint_values.name.index(joints[i].name)
			q = joint_values.position[j]
			axis = joints[i].axis
			T3 =tf.transformations.rotation_matrix(q, axis)
		else:
			T3 = tf.transformations.identity_matrix()

		T = tf.transformations.concatenate_matrices(T,T1,T2,T3)
		msg = message_from_transform(T,'world_link',link_names[i])
		br.sendTransform(msg)

	
def callback(joint_values):
	#rospy.loginfo("this is joint_values.name")
	#rospy.loginfo(joint_values.name)

	robot = URDF.from_parameter_server()

	link_names = []
        joints = []
	joints_names = []

	link_name = robot.get_root()
	while True:
            if link_name not in robot.child_map:
                break

            (next_joint_name, next_link_name) = robot.child_map[link_name][0]

            if next_joint_name not in robot.joint_map:
                break;

            joints.append(robot.joint_map[next_joint_name])
            link_names.append(next_link_name)
	    joints_names.append(next_joint_name)

            link_name = next_link_name

	#rospy.loginfo("this is joints_names")
	#rospy.loginfo(joints_names)

	calculate_and_publish_transforms(link_names, joints, joint_values)
	

def listener():
	rospy.init_node('whatever', anonymous=True)
    	rospy.Subscriber("joint_states", JointState, callback)
	
	rospy.spin()

if __name__ == '__main__':
	listener()
#rosrun asgn2 forward_kinematics.py

