#!/usr/bin/env python3
"""
DeepSeek 语音对话 Agent - 固定时长录音版
录音策略完全复刻 diag_mic.py 诊断脚本（已验证准确识别）
"""
import os
import sys
import json
import time
import queue
import threading
import pyaudio
import numpy as np
from vosk import Model, KaldiRecognizer
from openai import OpenAI

# 屏蔽 ALSA 噪音
sys.stderr = open('/tmp/alsa_err.log', 'w')

# 加载 ROS2 环境（如果可用）
ros2_setup = '/opt/ros/humble/setup.py'
if os.path.exists(ros2_setup):
    sys.path.insert(0, '/opt/ros/humble/lib/python3.10/site-packages')
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import Joy
        ROS2_AVAILABLE = True
    except ImportError:
        ROS2_AVAILABLE = False
else:
    ROS2_AVAILABLE = False

# 导入 audio_player
sys.path.append('/root/face_project/bpu_nodes')
from audio_player import AudioPlayer

# ==================== 配置 ====================
DEEPSEEK_API_KEY = "sk-4c2a9530abab4cf1a2d5a07eecb486ac"
MODEL_PATH = "/root/vosk_model/cn"
SAMPLE_RATE = 16000
CHUNK = 1024
# 音量 VAD（用于触发，阈值调高避免误触发）
SILENCE_THRESHOLD = 300      # 触发阈值
VAD_CONFIRM_FRAMES = 12      # 连续确认帧数（约0.8秒）
RECORD_SECONDS = 2.5         # VAD确认后继续录2.5秒（缩短延迟）
WAKE_WORD_SECONDS = 1.2      # 唤醒词检测录音时长（缩短）
# 唤醒词专用 VAD（更敏感，更快触发）
WAKE_VAD_FRAMES = 8          # 唤醒词确认帧数（约0.5秒，更快触发）
WAKE_THRESHOLD = 250         # 唤醒词阈值（更低，更敏感）
VOLUME_LEVEL = 220
# =============================================

class DeepSeekVoiceAgent:
    def __init__(self):
        # 恢复 stderr 用于正常打印（ALSA 噪音已重定向到文件）
        sys.stderr = sys.__stderr__
        print("🔄 加载 Vosk 模型...")
        self.model = Model(MODEL_PATH)

        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        self.p = pyaudio.PyAudio()

        # 打印可用输入设备，排查 systemd 环境下设备索引变化
        print("📋 可用音频输入设备:")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                print(f"  [{i}] {info.get('name')} (ch={info.get('maxInputChannels')}, rate={info.get('defaultSampleRate')})")

        # 确认 input_device_index=1 存在，否则回退到默认设备
        try:
            info = self.p.get_device_info_by_index(1)
            self.input_device = 1
            print(f"✅ 使用输入设备 [1]: {info.get('name')}")
        except Exception:
            self.input_device = None
            print("⚠️ 设备[1]不存在，使用系统默认输入设备")

        self.tts = AudioPlayer(volume=VOLUME_LEVEL)
        
        self.conversation_history = []

        # 初始化 ROS2 运动控制
        self.ros2_pub = None
        self.ros2_node = None
        self._init_ros2()

        print("🎤 语音助手启动")
        print("=" * 50)

    def record_and_recognize(self):
        """
        录音 + 识别（优化版：边录边识别 + 静音早停 + 缩短缓冲）
        1. 音量 VAD 触发（高阈值 + 严格确认，防误触发）
        2. 触发后边录边把每帧喂给 recognizer，处理与录音时间重叠
        3. 检测到说完（0.77秒静音）即提前结束，不再死等满 RECORD_SECONDS
        """
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=self.input_device
        )

        print("🔴 监听中...（等待说话）")

        # ===== 第1阶段：VAD 等待说话（滑动窗口，允许句中短暂停顿）=====
        confirm_count = 0
        frames = []  # 保留最近音频
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            volume = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
            if volume > SILENCE_THRESHOLD:
                confirm_count += 1
                frames.append(data)
                if confirm_count >= VAD_CONFIRM_FRAMES:
                    print("🎤 检测到说话，录音中...")
                    break
            else:
                # 不重置！允许短暂停顿（低音量只减1，不归零）
                confirm_count = max(0, confirm_count - 1)
                frames.append(data)
                # 缩短缓冲：只保留最近 16 帧（约1秒，含语音开头即可）
                if len(frames) > 16:
                    frames = frames[-16:]

        # ===== 第2阶段：快速录音（只做I/O，不识别）+ 静音早停 =====
        # 注意：本板 Vosk 处理比实时慢，边录边喂会丢帧，必须先读全再识别
        t0 = time.time()
        max_frames = int(SAMPLE_RATE / CHUNK * RECORD_SECONDS)
        min_frames = 16                # 至少录 ~1秒才允许早停
        silence_stop = 12              # 连续 ~0.77秒静音 → 提前结束
        silence_count = 0
        for i in range(max_frames):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            if i >= min_frames:
                vol = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
                if vol < SILENCE_THRESHOLD:
                    silence_count += 1
                    if silence_count >= silence_stop:
                        break
                else:
                    silence_count = 0

        stream.stop_stream()
        stream.close()

        # ===== 第3阶段：统一识别 =====
        t1 = time.time()
        recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        recognizer.SetWords(True)
        for frame in frames:
            recognizer.AcceptWaveform(frame)
        final_result = json.loads(recognizer.FinalResult())
        t2 = time.time()
        print(f"⏱️  录音{t1-t0:.1f}s 识别{t2-t1:.1f}s")

        text = final_result.get('text', '').strip()
        if text:
            print(f"📝 识别结果: {text}")
            return text
        print("📝 识别结果: (空)")
        return None

    def ask_deepseek(self, question):
        """DeepSeek API（非流式，完整返回，带工具调用）"""
        if "重置" in question or "重新开始" in question:
            self.conversation_history = []
            return "已重置"

        print(f"💭 问题: {question}")
        answer = ""
        for delta in self.ask_deepseek_stream(question):
            answer += delta
        return answer

    # ==================== 工具调用（Function Calling） ====================
    # 节点名称 → systemd 服务名
    NODE_SERVICES = {
        "相机": "vision-camera.service",
        "视觉": "vision-recognizer.service",
        "识别": "vision-recognizer.service",
        "语音": "voice-agent.service",
        "推理": "robot-inference.service",
        "运动": "robot-inference.service",
        "机器人": "robot-inference.service",
    }

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "start_node",
                "description": "启动一个系统节点（如相机、人脸识别等）。当用户要求打开/启动某个功能时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_name": {
                            "type": "string",
                            "description": "要启动的节点中文名",
                            "enum": ["相机", "视觉", "识别", "语音", "推理", "运动", "机器人"]
                        }
                    },
                    "required": ["node_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "stop_node",
                "description": "关闭一个系统节点。当用户要求关闭/停止某个功能时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_name": {
                            "type": "string",
                            "description": "要关闭的节点中文名",
                            "enum": ["相机", "视觉", "识别", "语音", "推理", "运动", "机器人"]
                        }
                    },
                    "required": ["node_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_nodes",
                "description": "列出所有节点及其运行状态。当用户询问当前运行了什么/状态时调用。",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "robot_stand_up",
                "description": "让机器人从蹲下状态站起来。当用户说站起、站起来、起身时调用。",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "robot_squat",
                "description": "让机器人蹲下。当用户说蹲下、蹲下吧、休息时调用。",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "robot_enter_inference",
                "description": "让机器人进入推理模式（先站起再进入推理），之后机器人才可以走路。当用户说进入推理模式、准备走路、开始行动时调用。",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "move_robot",
                "description": "控制机器人运动（前进/后退/左转/右转/停止）。当用户要求机器人移动、走路、转向时调用。机器人必须先进入推理模式才能运动。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "description": "运动方向",
                            "enum": ["forward", "backward", "left", "right", "stop"]
                        },
                        "duration": {
                            "type": "number",
                            "description": "运动持续时间（秒），默认1.5秒，停止时忽略"
                        }
                    },
                    "required": ["direction"]
                }
            }
        }
    ]

    def execute_tool(self, name, args):
        """执行工具调用，返回结果字符串"""
        import subprocess as sp
        if name == "list_nodes":
            results = []
            for cn_name, svc in self.NODE_SERVICES.items():
                r = sp.run(["systemctl", "is-active", svc], capture_output=True, text=True)
                status = "运行中" if r.stdout.strip() == "active" else "已停止"
                results.append(f"{cn_name}({svc}): {status}")
            return "\n".join(results)

        if name == "robot_stand_up":
            return self.robot_stand_up()

        if name == "robot_squat":
            return self.robot_squat()

        if name == "robot_enter_inference":
            return self.robot_enter_inference()

        if name == "move_robot":
            direction = args.get("direction", "stop")
            duration = args.get("duration", 1.5)
            return self.move_robot(direction, duration=duration)

        node = args.get("node_name", "")
        svc = self.NODE_SERVICES.get(node)
        if not svc:
            return f"未知节点: {node}"

        action = "start" if name == "start_node" else "stop"
        action_cn = "启动" if name == "start_node" else "关闭"
        try:
            sp.run(["systemctl", action, svc], capture_output=True, timeout=15)
            return f"已{action_cn}节点: {node}（{svc}）"
        except Exception as e:
            return f"{action_cn}{node}失败: {e}"

    def _init_ros2(self):
        """初始化 ROS2 节点和发布器"""
        if not ROS2_AVAILABLE:
            self.ros2_pub = None
            self.joy_pub = None
            self.ros2_node = None
            print("⚠️ ROS2 不可用，运动控制禁用")
            return False
        try:
            rclpy.init()
            self.ros2_node = Node("voice_cmd_vel")
            self.ros2_pub = self.ros2_node.create_publisher(Twist, "/cmd_vel", 10)
            self.joy_pub = self.ros2_node.create_publisher(Joy, "/joy", 10)
            self.ros2_thread = threading.Thread(target=self._ros2_spin, daemon=True)
            self.ros2_thread.start()
            self.robot_state = "unknown"
            print("✅ ROS2 cmd_vel + joy 发布器已启动")
            return True
        except Exception as e:
            print(f"⚠️ ROS2 初始化失败: {e}")
            self.ros2_pub = None
            self.joy_pub = None
            return False

    def _publish_joy_button(self, button_idx, hold_time=0.3):
        """模拟按下一个手柄按键（按下hold_time秒后松开）"""
        if not self.joy_pub:
            return
        # 按下
        msg = Joy()
        msg.header.stamp = self.ros2_node.get_clock().now().to_msg()
        msg.buttons = [0] * 12
        msg.axes = [0.0] * 6
        msg.buttons[button_idx] = 1
        # 以 20Hz 发布 hold_time 秒
        rate = 20
        for _ in range(int(hold_time * rate)):
            self.joy_pub.publish(msg)
            time.sleep(1.0 / rate)
        # 松开
        msg.buttons[button_idx] = 0
        for _ in range(5):
            self.joy_pub.publish(msg)
            time.sleep(0.05)

    def press_a(self):
        """按A键：站起/蹲下切换"""
        self._publish_joy_button(0)
        time.sleep(2.5)  # 等待过渡完成

    def press_b(self):
        """按B键：进入/退出推理模式"""
        self._publish_joy_button(1)
        time.sleep(1.5)

    def robot_stand_up(self):
        """机器人站起（从蹲下到站起）"""
        if not self.joy_pub:
            return "运动控制不可用"
        self.press_a()
        self.robot_state = "stand"
        return "机器人已站起"

    def robot_squat(self):
        """机器人蹲下"""
        if not self.joy_pub:
            return "运动控制不可用"
        self.press_a()
        self.robot_state = "squat"
        return "机器人已蹲下"

    def robot_enter_inference(self):
        """进入推理模式（先站起，再按B进入推理）"""
        if not self.joy_pub:
            return "运动控制不可用"
        if self.robot_state not in ("stand", "inference"):
            self.press_a()  # 先站起
        self.press_b()  # 按B进入推理
        self.robot_state = "inference"
        return "机器人已进入推理模式，可以行走了"

    def _ros2_spin(self):
        """ROS2 spin 后台线程"""
        if self.ros2_node:
            rclpy.spin(self.ros2_node)

    def move_robot(self, direction, duration=1.5, speed=0.3, turn_speed=0.5):
        """控制机器人运动
        direction: forward/backward/left/right/stop
        duration: 运动持续时间（秒）
        """
        if not self.ros2_pub:
            return "运动控制不可用，ROS2未初始化"

        if direction == "stop":
            stop_msg = Twist()
            self.ros2_pub.publish(stop_msg)
            return "机器人已停止"

        # 自动进入推理模式（如果还没进）
        if self.robot_state != "inference":
            self.robot_enter_inference()
            time.sleep(0.5)

        msg = Twist()
        if direction == "forward":
            msg.linear.x = speed
            action = "前进"
        elif direction == "backward":
            msg.linear.x = -speed
            action = "后退"
        elif direction == "left":
            msg.angular.z = turn_speed
            action = "左转"
        elif direction == "right":
            msg.angular.z = -turn_speed
            action = "右转"
        else:
            return f"未知方向: {direction}"

        # 以 10Hz 频率发布持续 duration 秒
        rate_hz = 10
        for _ in range(int(duration * rate_hz)):
            self.ros2_pub.publish(msg)
            time.sleep(1.0 / rate_hz)
        # 停止
        stop_msg = Twist()
        self.ros2_pub.publish(stop_msg)
        return f"机器人{action}{duration}秒"

    # ====================================================================

    def ask_deepseek_stream(self, question):
        """DeepSeek 流式 API - 生成器，逐字返回文本
        支持 Function Calling：先非流式检测是否需要调用工具，
        若需要则执行工具后再流式生成回复。
        """
        if "重置" in question or "重新开始" in question:
            self.conversation_history = []
            yield "已重置"
            return

        print(f"💭 问题: {question}")
        system_prompt = (
            "你是双足机器人的语音助手，名字叫小地。简洁回答，不超过30字。"
            "你可以通过工具控制机器人的节点（相机、视觉识别、推理节点等）。"
            "你还可以控制机器人状态：站起、蹲下、进入推理模式、运动。"
            "当用户要求打开/关闭/查询功能时，调用节点控制工具。"
            "当用户说站起/蹲下/进入推理模式时，调用对应的机器人状态工具。"
            "当用户要求机器人移动、走路、转向时，调用运动控制工具。"
            "机器人运动流程：先站起 → 进入推理模式 → 才能走路。"
            "运动默认持续1.5秒，除非用户明确说走多久。"
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history[-6:])
        messages.append({"role": "user", "content": question})

        try:
            # 第1次请求：非流式，带工具定义，判断是否需要调用工具
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=self.TOOLS,
                temperature=0.3,
                max_tokens=80
            )
            msg = resp.choices[0].message

            # 如果需要调用工具
            if msg.tool_calls:
                messages.append(msg)
                for tool_call in msg.tool_calls:
                    fname = tool_call.function.name
                    import json as _json
                    fargs = _json.loads(tool_call.function.arguments or "{}")
                    print(f"🔧 调用工具: {fname}({fargs})")
                    result = self.execute_tool(fname, fargs)
                    print(f"   → {result}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                # 第2次请求：带工具执行结果，流式生成自然语言回复
                stream = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=True,
                    temperature=0.5,
                    max_tokens=80
                )
                full_text = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        full_text += delta
                        yield delta
                self.conversation_history.append({"role": "user", "content": question})
                self.conversation_history.append({"role": "assistant", "content": full_text})
                print(f"\n🤖: {full_text}")
            else:
                # 不需要工具，直接回复
                answer = msg.content or ""
                self.conversation_history.append({"role": "user", "content": question})
                self.conversation_history.append({"role": "assistant", "content": answer})
                print(f"🤖: {answer}")
                yield answer
        except Exception as e:
            print(f"❌ API 失败: {e}")
            yield "抱歉，我无法回答"

    def speak_stream(self, text_generator):
        """流式 TTS 播放：逐句合成、逐句播放
        从 text_generator 获取文本流，按句子切分，一边合成一边播放
        """
        tts_queue = queue.Queue()      # 待合成的句子
        play_queue = queue.Queue()     # 待播放的 MP3 文件
        full_text = [""]               # 用列表实现闭包内可变

        def tts_worker():
            """TTS 合成 + 转码线程：合成 MP3 后转 WAV，放入 play_queue"""
            chunk_id = 0
            while True:
                sentence = tts_queue.get()
                if sentence is None:
                    play_queue.put(None)
                    break
                wav_file = f"/tmp/tts_chunk_{chunk_id}.wav"
                mp3_file = f"/tmp/tts_chunk_{chunk_id}.mp3"
                chunk_id += 1
                try:
                    import edge_tts
                    import subprocess
                    tts = edge_tts.Communicate(sentence, voice="zh-CN-XiaoxiaoNeural")
                    tts.save_sync(mp3_file)
                    # 转 WAV（用 plughw 播放，和视觉模块兼容共享声卡）
                    cmd = f'ffmpeg -y -i {mp3_file} -ac 2 -ar 48000 {wav_file} 2>/dev/null'
                    subprocess.run(cmd, shell=True, check=True)
                    play_queue.put(wav_file)
                except Exception as e:
                    print(f"❌ TTS 合成失败: {e}")
                    play_queue.put(None)

        def play_worker():
            """播放线程：从 play_queue 取 WAV 文件用 aplay 播放"""
            import subprocess
            while True:
                wav_file = play_queue.get()
                if wav_file is None:
                    break
                try:
                    cmd = f'aplay -D plughw:0,0 {wav_file} 2>/dev/null'
                    subprocess.run(cmd, shell=True, check=True)
                except Exception as e:
                    print(f"❌ 播放失败: {e}")

        # 启动两个后台线程
        threading.Thread(target=tts_worker, daemon=True).start()
        play_thread = threading.Thread(target=play_worker, daemon=True)
        play_thread.start()

        # 收集文本流，按句子切分送入 TTS 队列
        sentence_buf = ""
        for delta in text_generator:
            full_text[0] += delta
            sentence_buf += delta
            # 遇到句末标点就切分
            for end_char in ['。', '！', '？', '\n', '；']:
                if end_char in sentence_buf:
                    idx = sentence_buf.index(end_char) + 1
                    sentence = sentence_buf[:idx].strip()
                    sentence_buf = sentence_buf[idx:]
                    if sentence:
                        tts_queue.put(sentence)
                    break

        # 最后不足一句的剩余部分也送进去
        if sentence_buf.strip():
            tts_queue.put(sentence_buf.strip())

        # 通知 TTS 线程结束
        tts_queue.put(None)

        # 等待播放线程结束（play_worker 收到 None 后退出）
        play_thread.join()

        return full_text[0]

    def wait_for_wake_word(self):
        """等待唤醒词"你好小地"，检测到后回复"我在"并返回 True"""
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=self.input_device
        )

        print("🔴 等待唤醒词（说'你好小地'）...")

        # 唤醒词专用 VAD：更低阈值 + 更少确认帧 → 更敏感更快触发
        confirm_count = 0
        frames = []
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            volume = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
            if volume > WAKE_THRESHOLD:
                confirm_count += 1
                frames.append(data)
                if confirm_count >= WAKE_VAD_FRAMES:
                    break
            else:
                confirm_count = max(0, confirm_count - 1)
                frames.append(data)
                if len(frames) > 40:
                    frames = frames[-40:]

        # 录 WAKE_WORD_SECONDS 秒
        frames_to_record = int(SAMPLE_RATE / CHUNK * WAKE_WORD_SECONDS)
        for i in range(frames_to_record):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        # 识别
        recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        recognizer.SetWords(True)
        for frame in frames:
            recognizer.AcceptWaveform(frame)

        text = json.loads(recognizer.FinalResult()).get('text', '').strip()
        # Vosk 输出带空格（如"你好 小弟"），去掉空格再做匹配
        text_no_space = text.replace(" ", "")
        print(f"📝 唤醒词识别: {text}")

        if "你好小地" in text_no_space or "你好小弟" in text_no_space:
            print("🔔 唤醒成功！")
            self.speak("我在，今天过得开心吗")
            return True
        return False

    def speak(self, text):
        print("🔊 播报")
        self.tts.say(text)

    

    def run(self):
        try:
            while True:
                # 第1步：等待唤醒词
                while not self.wait_for_wake_word():
                    time.sleep(0.1)

                # 第2步：听问题
                text = self.record_and_recognize()
                if text:
                    if text in ['再见', '退出']:
                        print("👋 再见")
                        break
                    if "你好小地" in text.replace(" ", "") or "你好小弟" in text.replace(" ", ""):
                        continue
                    # 流式响应 + 流式 TTS 播放，大幅降低感知延迟
                    t0 = time.time()
                    print("⚡ 流式响应中...")
                    self.speak_stream(self.ask_deepseek_stream(text))
                    print(f"⏱️  总耗时: {time.time()-t0:.1f}s")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 退出")
        finally:
            self.p.terminate()

if __name__ == "__main__":
    agent = DeepSeekVoiceAgent()
    agent.run()
