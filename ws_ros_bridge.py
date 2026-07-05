#!/usr/bin/env python3
"""
Tinker RL — WebSocket → ROS2 命令桥接节点
从 /tmp/robot_cmd.json 读取前端命令，发布为 ROS2 话题

映射说明 (飞智沙漠狐手柄):
  前端按钮 → Joy消息 → inference_node状态机
───────────────────────────────────────────
  启动  → Joy.buttons[1]=1 (B键) → 进入推理模式
  停止  → Joy.buttons[1]=1 (B键) → 退出推理模式
  站立  → Joy.buttons[0]=1 (A键) → 切换到站立
  行走  → Joy.buttons[1]=1 (B键) → 进入推理(行走)
  急停  → 全部关节置零 + 停止推理
  左摇杆 → geometry_msgs/Twist → cmd_vel

启动:
  ros2 run robot_control ws_ros_bridge.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import json
import os
import time

CMD_FILE = "/tmp/robot_cmd.json"
CMD_VEL_FILE = "/tmp/robot_cmd_vel.json"  # 速度指令单独文件（高频写入）


class WsRosBridge(Node):
    def __init__(self):
        super().__init__('ws_ros_bridge')

        # 发布器
        self.joy_pub = self.create_publisher(Joy, '/joy', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.motor_pub = self.create_publisher(Float64MultiArray, 'motor_cmds', 10)

        # 上次处理的命令
        self.last_cmd_ts = 0
        self.last_vel_ts = 0

        # 状态
        self.last_joy = Joy()
        self.last_joy.buttons = [0] * 12
        self.last_joy.axes = [0.0] * 8

        # 50Hz 循环检查命令
        self.timer = self.create_timer(0.02, self.timer_callback)
        self.get_logger().info('WebSocket → ROS2 桥接已启动')

    def timer_callback(self):
        now = time.time()

        # ─── 检查控制命令 ───
        cmd = self._read_json(CMD_FILE)
        if cmd and cmd.get('timestamp', 0) > self.last_cmd_ts:
            self.last_cmd_ts = cmd['timestamp']
            self._handle_command(cmd)

        # ─── 检查速度指令 ───
        vel = self._read_json(CMD_VEL_FILE)
        if vel and vel.get('timestamp', 0) > self.last_vel_ts:
            self.last_vel_ts = vel['timestamp']
            self._handle_velocity(vel)

    def _read_json(self, path):
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except:
            pass
        return None

    def _send_joy_click(self, button_idx, duration=0.1):
        """模拟一次手柄按键点击"""
        # 按下
        joy = Joy()
        joy.header.stamp = self.get_clock().now().to_msg()
        joy.buttons = [0] * 12
        joy.buttons[button_idx] = 1
        joy.axes = [0.0] * 8
        self.joy_pub.publish(joy)

        # 松开（用timer延迟松开）
        self.create_timer(duration, lambda: self._release_joy())

    def _release_joy(self):
        joy = Joy()
        joy.header.stamp = self.get_clock().now().to_msg()
        joy.buttons = [0] * 12
        joy.axes = [0.0] * 8
        self.joy_pub.publish(joy)

    def _handle_command(self, cmd):
        cmd_type = cmd.get('command', '')
        self.get_logger().info(f'收到命令: {cmd_type}')

        if cmd_type == 'start':
            # 模拟B键 → 进入推理模式
            self._send_joy_click(1)
            self.get_logger().info('▶ 启动推理')

        elif cmd_type == 'stop':
            # 模拟B键 → 退出推理模式
            self._send_joy_click(1)
            self.get_logger().info('■ 停止推理')

        elif cmd_type == 'mode_switch':
            mode = cmd.get('mode', 'stand')
            if mode == 'stand':
                # 模拟A键 → 切换到站立
                self._send_joy_click(0)
                self.get_logger().info('🧍 切换到站立模式')
            elif mode == 'walk':
                # 先B键进入推理
                self._send_joy_click(1)
                self.get_logger().info('🚶 切换到行走模式')

        elif cmd_type == 'emergency_stop':
            # 紧急停止: 发送零位关节指令 + 停止推理
            self.get_logger().warn('⚠ 紧急停止!')

            # 1. 发送零位关节
            zero_msg = Float64MultiArray()
            zero_msg.data = [0.0] * 10
            self.motor_pub.publish(zero_msg)

            # 2. 停止速度
            twist = Twist()
            self.cmd_vel_pub.publish(twist)

            # 3. 模拟B键退出推理
            self._send_joy_click(1)

        elif cmd_type == 'calibrate_imu':
            self.get_logger().info('校准 IMU (无实际动作)')

        elif cmd_type == 'reset_odom':
            self.get_logger().info('重置里程 (无实际动作)')

    def _handle_velocity(self, vel):
        """处理速度指令（虚拟摇杆）"""
        twist = Twist()
        twist.linear.x = float(vel.get('vx', 0.0))
        twist.linear.y = float(vel.get('vy', 0.0))
        twist.angular.z = float(vel.get('dyaw', 0.0))
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = WsRosBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
