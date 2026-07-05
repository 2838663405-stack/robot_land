#!/usr/bin/env python3
"""
Tinker RL — 摄像头 HTTP 推流节点
把 ROS2 图像话题转为 JPEG，通过 HTTP 供前端显示

启动:
  source /home/root/legged-robot/install/setup.bash
  python3 /root/camera_http_server.py

前端访问:
  http://192.168.50.195:8080/camera.jpg
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import numpy as np
import cv2

latest_frame = b''

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_http_server')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.get_logger().info('摄像头订阅已启动')

    def image_callback(self, msg):
        global latest_frame
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            # 压缩为 JPEG
            _, jpeg = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
            latest_frame = jpeg.tobytes()
        except Exception as e:
            self.get_logger().error(f'图像处理错误: {e}')


class CameraHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/camera.jpg':
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            if latest_frame:
                self.wfile.write(latest_frame)
            else:
                # 返回一个占位图
                self.wfile.write(b'')
        else:
            self.send_response(404)
            self.end_headers()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()

    # 启动 HTTP 服务器（端口 8080）
    httpd = HTTPServer(('0.0.0.0', 8080), CameraHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    node.get_logger().info('摄像头 HTTP 服务已启动: http://0.0.0.0:8080/camera.jpg')

    rclpy.spin(node)
    httpd.shutdown()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
