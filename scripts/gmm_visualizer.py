#!/usr/bin/env python3


import rospy
import sys
import math
from geometry_msgs.msg import Point
from gmm_msgs.msg import GMM, Gaussian
from visualization_msgs.msg import Marker, MarkerArray


class GMM_Vis():
    def __init__(self):
        rospy.init_node("gmm_visualizer")
        
        pub = rospy.Publisher("/gmm_visualizer", MarkerArray, queue_size=1)       
        sub = rospy.Subscriber("/gaussian_mixture_model", GMM, self.callback)

        self.pub = pub
        self.sub = sub
        
        self.gmm_msg = GMM()
        self.marker_msg = MarkerArray()

    def callback(self, msg):
        self.gmm_msg.gaussians = msg.gaussians
        self.gmm_msg.weights = msg.weights

    def GroundColorMix(self, x, min = 0.0, max = 1.0):
        posSlope = (max-min)/60
        negSlope = (min-max)/60
        color = [0.0, 0.0, 0.0]

        if x < 60:
            color[0] = max
            color[1] = posSlope + min
            color[2] = min
            return color
        elif x < 120:
            color[0] = negSlope * x + 2 * max + min
            color[1] = max
            color[2] = min
            return color
        elif x < 180:
            color[0] = min
            color[1] = max
            color[2] = posSlope * x - 2 * max + min
            return color
        elif x < 240:
            color[0] = min
            color[1] = negSlope * x + 4 * max + min
            color[2] = max
            return color
        elif x < 300:
            color[0] = posSlope * x - 4 * max + min
            color[1] = min
            color[2] = max
            return color
        else:
            color[0] = max
            color[1] = min
            color[2] = negSlope * x + 6 * max
            return color
        
    def euler_to_quat(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr

        return [w, x, y, z]

        

    
    def show(self):
        for i in range(len(self.gmm_msg.weights)):
            # Draw ellipse from covariance matrix
            a = self.gmm_msg.gaussians[i].covariance[0]
            b = self.gmm_msg.gaussians[i].covariance[1]
            c = self.gmm_msg.gaussians[i].covariance[3]

            # Ellipses parameters
            lambda1 = (a+c)/2 + ((a-c)**2 + 4*b**2)**0.5/2
            lambda2 = (a+c)/2 - ((a-c)**2 + 4*b**2)**0.5/2
            r1 = lambda1**0.5
            r2 = lambda2**0.5
            theta = 0.0
            if b == 0.0 and a >= c:
                theta = 0.0
            elif b == 0.0 and a < c:
                theta = 0.5 * math.pi
            else:
                theta = math.atan2(lambda1 - a, b)

            m = Marker()
            m.id = i
            m.header.frame_id = "map"
            m.type = m.SPHERE
            m.action = m.ADD
            m.pose.position.x = self.gmm_msg.gaussians[i].mean_point.x
            m.pose.position.y = self.gmm_msg.gaussians[i].mean_point.y
            m.pose.position.z = 0.0

            m.scale.x = r1
            m.scale.y = r2
            m.scale.z = 0.1

            [qw, qx, qy, qz] = self.euler_to_quat(0.0, 0.0, theta)
            m.pose.orientation.w = qw
            m.pose.orientation.x = qx
            m.pose.orientation.y = qy
            m.pose.orientation.z = qz

            color = self.GroundColorMix(360.0 * self.gmm_msg.weights[i])
            m.color.r = color[0]
            m.color.g = color[1]
            m.color.b = color[2]
            m.color.a = 1.0

            self.marker_msg.markers.append(m)
            self.pub.publish(self.marker_msg)







def main():
    viz = GMM_Vis()
    
    while not rospy.is_shutdown():
        viz.show()
        rospy.sleep(1)
    # rospy.spin()


if __name__ == '__main__':
    main()
    


