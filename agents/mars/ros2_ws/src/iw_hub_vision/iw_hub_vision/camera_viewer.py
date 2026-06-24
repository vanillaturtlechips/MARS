import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        self.sub = self.create_subscription(
            Image, '/camera/image_raw', self.callback, 10)
        self.pub = self.create_publisher(Detection2DArray, '/detections', 10)

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(frame, verbose=False)

        det_array = Detection2DArray()
        det_array.header = msg.header

        for box in results[0].boxes:
            det = Detection2D()
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self.model.names[int(box.cls)]
            hyp.hypothesis.score = float(box.conf)
            det.results.append(hyp)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det.bbox.center.position.x = (x1 + x2) / 2
            det.bbox.center.position.y = (y1 + y2) / 2
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1
            det_array.detections.append(det)

        self.pub.publish(det_array)

        annotated = results[0].plot()
        cv2.imshow('camera', annotated)
        cv2.waitKey(1)


def main():
    rclpy.init()
    rclpy.spin(CameraViewer())


if __name__ == '__main__':
    main()
