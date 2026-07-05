#!/usr/bin/env python3
"""
Tinker RL Control System — RDK X5 WebSocket 数据服务器
在 RDK X5 上运行，读取 RL 推理数据并通过 WebSocket 发送到前端

用法:
  python3 rdk_ws_server.py [--port 8765] [--freq 50]

数据源接入方式（二选一）:
  方式1 (推荐): 从共享内存/文件读取 RL 推理结果
  方式2: 直接导入你的 RL 推理模块并调用

依赖安装:
  pip3 install websockets
"""
import asyncio
import json
import time
import math
import random
import argparse
import os
import struct

# ─── 可配置的数据源路径 ───
SHARED_MEM_PATH = "/tmp/tinker_telemetry.json"  # RL 推理写入的共享文件

# ─── 模拟数据 (当真实数据文件不存在时自动使用) ───
WALK_PATTERN = [
    (0.5, 0.8), (0.8, 0.6), (1.0, 0.5), (1.2, 0.4), (1.5, 0.7),
    (0.6, 0.6), (0.9, 0.5), (1.1, 0.4), (1.3, 0.5), (0.7, 0.3),
]


class MockDataGenerator:
    """无真实数据时，生成模拟数据让前端能看到动态效果"""
    def __init__(self):
        self.t = 0.0
        self.mode = "walk"

    def generate(self) -> dict:
        self.t += 0.02 + random.uniform(-0.005, 0.005)

        # IMU
        if self.mode == "walk":
            roll = 5.0 * math.sin(self.t * 1.5) + random.gauss(0, 0.3)
            pitch = 3.0 * math.sin(self.t * 2.0 + 0.5) + random.gauss(0, 0.3)
            yaw = 2.0 * math.sin(self.t * 0.8) + random.gauss(0, 0.3)
        elif self.mode == "stand":
            roll = random.gauss(0, 0.3)
            pitch = random.gauss(0, 0.3)
            yaw = random.gauss(0, 0.3)
        else:
            roll = 85.0 + random.gauss(0, 2.0)
            pitch = 30.0 + random.gauss(0, 1.0)
            yaw = 45.0 + random.gauss(0, 2.0)

        # Action
        if self.mode == "fall":
            action = [random.uniform(-0.5, 0.5) for _ in range(10)]
        else:
            pattern = WALK_PATTERN if self.mode == "walk" else [(0.1, 0.05)] * 10
            action = []
            for i, (freq, amp) in enumerate(pattern):
                val = amp * math.sin(self.t * freq * math.pi + i * 0.3)
                val += random.gauss(0, 0.02)
                action.append(round(max(-1.0, min(1.0, val)), 4))

        # Reward
        if self.mode == "walk":
            reward = 0.65 + 0.2 * math.sin(self.t * 0.3) + random.gauss(0, 0.05)
        elif self.mode == "stand":
            reward = 0.85 + 0.05 * math.sin(self.t * 0.2) + random.gauss(0, 0.05)
        else:
            reward = -0.5 + random.gauss(0, 0.1)
        reward = max(-1.0, min(1.0, reward))

        return {
            "imu": {"roll": round(roll, 2), "pitch": round(pitch, 2), "yaw": round(yaw, 2)},
            "reward": round(reward, 4),
            "action": action,
            "battery": round(12.3 - random.random() * 0.05, 3),
            "mode": self.mode,
            "status": {"uptime": int(time.time()), "is_standing": self.mode != "fall", "is_moving": self.mode == "walk", "is_fallen": self.mode == "fall", "control_freq": round(50 + random.uniform(-3, 3), 1)},
            "system": {"cpu_load": round(35 + 15 * random.random(), 1), "memory_usage": round(40 + 30 * random.random(), 0), "temperature": round(40 + 10 * random.random(), 1), "fps": round(50 + random.uniform(-3, 3), 1)},
        }


class RLDataBridge:
    """
    RL 推理数据桥接器
    ─────────────────────────────────────────────
    你需要将你的 RL 推理结果写入以下任一位置：

    方式 A: 写入 JSON 文件 → /tmp/tinker_telemetry.json
    方式 B: 直接调用 self.update(data) 方法

    JSON 格式:
    {
        "imu": {"roll": 0.5, "pitch": -2.3, "yaw": 15.0},
        "reward": 0.85,
        "action": [0.1, -0.2, 0.3, ...],   # 10个关节值
        "battery": 12.3,
        "mode": "walk",                     # stand / walk / fall
        "status": {
            "uptime": 3600,
            "is_standing": true,
            "is_moving": true,
            "is_fallen": false,
            "control_freq": 50.0
        },
        "system": {
            "cpu_load": 45.2,
            "memory_usage": 65,
            "temperature": 42.5,
            "fps": 50.0
        }
    }
    """

    def __init__(self):
        self._data = {}
        self._mode = "stand"
        self._start_time = time.time()
        self._mock = MockDataGenerator()
        self._use_mock = False  # 有真实文件后自动切换

    def update(self, data: dict):
        """方式 B: 在你的 RL 推理循环中直接调用此方法"""
        self._data.update(data)
        self._data["timestamp"] = int(time.time() * 1000)
        if "mode" in data:
            self._mode = data["mode"]

    def get_telemetry(self) -> dict:
        """获取最新遥测数据"""
        # 尝试从文件读取 (方式 A)
        if os.path.exists(SHARED_MEM_PATH):
            self._use_mock = False
            try:
                with open(SHARED_MEM_PATH, 'r') as f:
                    file_data = json.load(f)
                    self._data.update(file_data)
            except (json.JSONDecodeError, IOError):
                pass

        # 没有真实数据 → 使用模拟数据
        if not self._data:
            self._use_mock = True
            mock = self._mock.generate()
            mock["mode"] = self._mode
            return mock

        # 真实数据：更新时间戳和运行时长
        self._data["timestamp"] = int(time.time() * 1000)
        if "status" not in self._data or not isinstance(self._data["status"], dict):
            self._data["status"] = {}
        self._data["status"]["uptime"] = int(time.time() - self._start_time)

        return self._data

    def handle_command(self, cmd: dict) -> str:
        """处理从前端收到的控制命令"""
        cmd_type = cmd.get("command", "")
        cmd["timestamp"] = time.time()

        # 转发命令到 ROS2 桥接
        try:
            with open("/tmp/robot_cmd.json", "w") as f:
                json.dump(cmd, f)
        except:
            pass

        if cmd_type == "start":
            self._mode = "stand"
            self._mock.mode = "stand"
            return "系统启动 → 站立模式"

        elif cmd_type == "stop":
            self._mode = "unknown"
            self._mock.mode = "unknown"
            return "系统停止"

        elif cmd_type == "mode_switch":
            new_mode = cmd.get("mode", "stand")
            if new_mode in ("stand", "walk"):
                self._mode = new_mode
                self._mock.mode = new_mode
                self._data["mode"] = new_mode
                return f"模式切换 → {'行走' if new_mode == 'walk' else '站立'}"
            return f"未知模式: {new_mode}"

        elif cmd_type == "emergency_stop":
            self._mode = "fall"
            self._mock.mode = "fall"
            self._data["mode"] = "fall"
            return "⚠ 紧急停止已触发"

        elif cmd_type == "calibrate_imu":
            return "IMU 校准已启动"

        elif cmd_type == "reset_odom":
            return "里程计已重置"

        elif cmd_type == "ping":
            return None  # 心跳，不回复

        elif cmd_type == "set_vel":
            try:
                with open("/tmp/robot_cmd_vel.json", "w") as f:
                    json.dump(cmd, f)
                return f"速度设置: vx={cmd.get('vx',0):.2f}"
            except:
                return "速度设置失败"

        return f"未知命令: {cmd_type}"

        return f"未知命令: {cmd_type}"


# ═══════════════════════════════════════════════════════════════
# WebSocket Server
# ═══════════════════════════════════════════════════════════════

async def handler(websocket):
    """处理单个前端连接"""
    addr = websocket.remote_address
    print(f"[+] 前端已连接: {addr}")
    bridge = RLDataBridge()
    send_count = 0

    try:
        while True:
            # ── 检查是否有命令 ──
            try:
                message = await asyncio.wait_for(
                    websocket.recv(), timeout=1.0 / args.freq
                )
                data = json.loads(message)
                response = bridge.handle_command(data)
                if response is not None:
                    await websocket.send(json.dumps({
                        "type": "command_response",
                        "message": response,
                    }))
                    print(f"[<] 命令: {data.get('command')} → {response}")
                continue  # 跳过本次数据发送，保证响应及时
            except asyncio.TimeoutError:
                pass  # 无命令，继续发送数据

            # ── 发送遥测数据 ──
            telemetry = bridge.get_telemetry()
            await websocket.send(json.dumps(telemetry))
            send_count += 1

            # 每 100 帧打印一次状态
            if send_count % 100 == 0:
                t = telemetry
                print(f"[→] 已发送 {send_count} 帧 | "
                      f"mode={t.get('mode','?')} | "
                      f"reward={t.get('reward',0):.3f} | "
                      f"bat={t.get('battery',0):.1f}V")

            await asyncio.sleep(1.0 / args.freq)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[!] 错误: {e}")
    finally:
        print(f"[-] 前端已断开: {addr} (共发送 {send_count} 帧)")


async def main():
    import websockets.server

    print(f"""
╔══════════════════════════════════════════════╗
║  Tinker RL · RDK X5 WebSocket Server       ║
║  ─────────────────────────────────────────── ║
║  监听地址:  ws://0.0.0.0:{args.port}             ║
║  发送频率:  {args.freq} Hz                      ║
║  数据来源:  {SHARED_MEM_PATH}                 ║
║                                              ║
║  在前端 monitor.html 中修改:                 ║
║  const wsUrl = 'ws://<RDK_X5_IP>:{args.port}';  ║
╚══════════════════════════════════════════════╝
    """)

    async with websockets.server.serve(handler, "0.0.0.0", args.port):
        await asyncio.Future()  # 永久运行


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tinker RDK X5 WebSocket Server")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket 端口")
    parser.add_argument("--freq", type=int, default=50, help="数据发送频率 (Hz)")
    args = parser.parse_args()

    asyncio.run(main())
