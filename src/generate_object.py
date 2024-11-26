#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Vector3

def publish_object_shape():
    # 初始化节点
    rospy.init_node('object_shape_publisher', anonymous=True)
    # 创建一个发布者
    marker_pub = rospy.Publisher('/object_shape_marker', Marker, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    # 设置物体尺寸
    obj_scale = Vector3(1.6014001, 0.99931884, 1.2036798)

    while not rospy.is_shutdown():
        # 创建Marker消息
        marker = Marker()
        marker.header.frame_id = "camera_color_optical_frame"  # 在此设置合适的坐标系
        marker.header.stamp = rospy.get_rostime()
        marker.ns = "object"
        marker.id = 0
        marker.type = Marker.CUBE  # 使用CUBE来表示矩形形状
        marker.action = Marker.ADD

        # 设置物体位置和尺寸
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = obj_scale.z / 2.0  # 放置物体使其底面在 z=0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = obj_scale.x
        marker.scale.y = obj_scale.y
        marker.scale.z = obj_scale.z

        # 设置物体的颜色
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8  # 设置透明度

        # 发布Marker
        marker_pub.publish(marker)
        rospy.loginfo("Publishing object shape marker in RViz")

        # 延时
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_object_shape()
    except rospy.ROSInterruptException:
        pass
