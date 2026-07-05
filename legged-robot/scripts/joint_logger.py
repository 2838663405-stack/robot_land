#!/usr/bin/env python3
"""
订阅 /joint_states 话题，实时保存关节数据到 CSV，同时启动 HTTP 服务器供可视化页面拉取数据。

用法:
    ros2 run robot_control joint_logger          # 作为 ROS2 entry point
    python3 joint_logger.py                       # 直接运行
    python3 joint_logger.py --port 8080 --output /tmp/joints.csv

CSV 格式:
    timestamp, j0_pos, j0_vel, j1_pos, j1_vel, ..., j9_pos, j9_vel
"""

import csv
import os
import sys
import threading
import time
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# ── 配置 ──────────────────────────────────────────────
DEFAULT_PORT = 8080
CSV_HEADER = ["timestamp"] + [f"j{i}_{k}" for i in range(10) for k in ("pos", "vel")]
# 当前活跃的 CSV 路径，HTTP 服务器会 serve 这个文件
ACTIVE_CSV = Path("/tmp/joint_log_active.csv")
# 可视化 HTML 路径
VIEWER_HTML = Path(__file__).parent / "joint_viewer.html"


class JointLoggerNode(Node):
    def __init__(self, output_path: str):
        super().__init__("joint_logger")
        self.csv_path = Path(output_path)
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(CSV_HEADER)
        self.csv_file.flush()

        # 同步一份到固定路径供 HTTP 服务器读取
        try:
            if ACTIVE_CSV.exists() or ACTIVE_CSV.is_symlink():
                ACTIVE_CSV.unlink()
            ACTIVE_CSV.symlink_to(self.csv_path.resolve())
        except OSError:
            # 符号链接失败时退化为复制
            ACTIVE_CSV.write_text("")

        self.msg_count = 0
        self.start_time = time.time()

        self.create_subscription(JointState, "/joint_states", self._cb, 10)
        self.get_logger().info(f"Logging to: {self.csv_path}")
        self.get_logger().info(f"Active CSV symlink: {ACTIVE_CSV}")

    def _cb(self, msg: JointState):
        now = time.time() - self.start_time
        row = [f"{now:.4f}"]
        for i in range(10):
            pos = msg.position[i] if len(msg.position) > i else 0.0
            vel = msg.velocity[i] if len(msg.velocity) > i else 0.0
            row.append(f"{pos:.6f}")
            row.append(f"{vel:.6f}")
        self.writer.writerow(row)
        self.msg_count += 1

        # 每 50 条 flush 一次，平衡 I/O 性能和数据安全
        if self.msg_count % 50 == 0:
            self.csv_file.flush()
            # 如果用的是复制模式，同步更新
            if not ACTIVE_CSV.is_symlink():
                import shutil
                shutil.copy2(self.csv_path, ACTIVE_CSV)

    def destroy_node(self):
        self.csv_file.flush()
        self.csv_file.close()
        self.get_logger().info(f"Total messages logged: {self.msg_count}")
        super().destroy_node()


# ── HTTP 服务器 ───────────────────────────────────────
class QuietHandler(SimpleHTTPRequestHandler):
    """静默日志的 HTTP handler，额外提供 /data.csv 端点。"""

    def log_message(self, *args):
        pass  # 抑制访问日志

    def do_GET(self):
        if self.path in ("/data.csv", "/joint_log_active.csv"):
            self._serve_csv()
        elif self.path == "/" or self.path == "/index.html":
            self._serve_html()
        else:
            super().do_GET()

    def _serve_csv(self):
        try:
            data = ACTIVE_CSV.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self.send_error(404)

    def _serve_html(self):
        try:
            html = VIEWER_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.end_headers()
            self.wfile.write(html)
        except Exception:
            self.send_error(404)


def start_http_server(port: int):
    server = HTTPServer(("0.0.0.0", port), QuietHandler)
    server.serve_forever()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Joint states logger + HTTP viewer")
    parser.add_argument("--output", "-o", default=None, help="CSV output path")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT, help="HTTP server port")
    args, remaining = parser.parse_known_args()

    output_path = args.output or f"joint_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # 启动 HTTP 服务器（后台线程）
    http_thread = threading.Thread(target=start_http_server, args=(args.port,), daemon=True)
    http_thread.start()
    print(f"[HTTP] Viewer server: http://localhost:{args.port}/")

    # 启动 ROS2 节点
    rclpy.init(args=remaining)
    node = JointLoggerNode(output_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()