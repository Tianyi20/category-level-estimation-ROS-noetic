import rospy
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Point,Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from lib.opts import opts
from lib.detectors.detector_factory import detector_factory
import copy
import cv2
import numpy as np
class MultiModelROSNode:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node('multi_model_ros_node', anonymous=True)

        # 使用 CvBridge 将 ROS 图像消息转换为 OpenCV 格式
        self.bridge = CvBridge()

        # 订阅相机图像和相机内参话题
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)

        # 初始化相机内参和其他参数
        self.camera_matrix = None
        self.detectors = []
        self.meta = {}
        self.init_detectors()

        # 初始化位姿发布者
        self.cup_pose_pub = rospy.Publisher('/cup_pose', PoseStamped, queue_size=10)
        self.bottle_pose_pub = rospy.Publisher('/bottle_pose', PoseStamped, queue_size=10)

        # 用于检测帧的计数
        self.idx = 0

    def init_detectors(self):
        # 初始化第一个模型
        opt_1 = copy.deepcopy(opts().parser.parse_args())
        opt_1.debug = 5
        opt_1.nms = True
        opt_1.obj_scale = True
        opt_1.arch = "dlav1_34"
        opt_1.load_model = "/home/tianyi/pose_estimation/src/CenterPose/models/cup_mug_v1_140.pth"
        opt_1.use_pnp = True
        
        # 更新默认配置
        opt_1 = opts().parse(opt_1)
        opt_1 = opts().init(opt_1)

        # 创建第二个模型的配置
        opt_2 = copy.deepcopy(opt_1)
        opt_2.load_model = "/home/tianyi/pose_estimation/src/CenterPose/models/bottle_v1_sym_12_140.pth"

        #threshold for filtering
        self.threshold_cup = 0.6
        self.threshold_bottle = 0.7

        # 添加到 detectors 列表
        opt_multiple = [opt_1, opt_2]
        for opt in opt_multiple:
            Detector = detector_factory[opt.task]
            detector = Detector(opt)
            self.detectors.append(detector)

    def camera_info_callback(self, data):
        # 获取相机内参矩阵
        self.camera_matrix = np.array(data.K).reshape(3, 3)
        self.meta['camera_matrix'] = self.camera_matrix

    def image_callback(self, data):
        try:
            # 将 ROS 图像消息转换为 OpenCV 格式
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # 确保相机内参已获取
        if self.camera_matrix is None:
            rospy.logwarn("Camera matrix not yet received. Skipping frame.")
            return

        # 显示图像
        cv2.imshow('input', img)
        if cv2.waitKey(1) == 27:  # 按下 'Esc' 键退出
            rospy.signal_shutdown("User requested shutdown.")

        # 使用每个模型进行推理
        for i, detector in enumerate(self.detectors):
            filename = f"model_{i}_frame_{str(self.idx).zfill(4)}.png"
            ret = detector.run(img, meta_inp=self.meta, filename=filename)
            rospy.loginfo(f"Model {i} results: {ret['results']}")
            
            # 处理结果，提取位置信息和四元数，生成并发布位姿;#注意这里的results是一个字典储存在ret里，
            # 如果场景里有多个相同类型的物体，则会有多个results在net里，但是这里我们只考虑桌面上相同类型的物体只有一个
            results = ret['results']
            ## 首先判断results为不为空，如果为空就直接跳过
            if len(results) == 0:
                rospy.loginfo(f"Model {i} detected no objects. Skipping.")
                continue
            result = results[0] # 提取result数组的第一个元素，所以我们桌上的相同类型的物体只能有一个
            #再次判断score得分，这里的threshould
            if i == 0 and result.get('score') < self.threshold_cup :
                continue
            elif i == 1 and result.get('score') < self.threshold_bottle:
                continue
            location = result.get('location')
            if location is not None:
                location = [location[1]*0.1,location[0]*0.1,location[2]*-0.1]
            quaternion = result.get('quaternion_xyzw')
            if location is not None and quaternion is not None:
                pose_stamped = PoseStamped()
                pose_stamped.header = Header()
                pose_stamped.header.stamp = rospy.Time.now()
                pose_stamped.header.frame_id = "camera_color_optical_frame"  # 需要根据你的TF树来设置正确的 frame_id
                pose_stamped.pose.position = Point(*location)
                pose_stamped.pose.orientation = Quaternion(*quaternion)
                if i == 0:  # 模型 0 为杯子
                    self.cup_pose_pub.publish(pose_stamped)
                    rospy.loginfo(f"Published cup pose: {pose_stamped}")
                elif i == 1:  # 模型 1 为瓶子
                    self.bottle_pose_pub.publish(pose_stamped)
                    rospy.loginfo(f"Published bottle pose: {pose_stamped}")

        # 增加帧计数
        self.idx += 1

if __name__ == '__main__':
    try:
        node = MultiModelROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
