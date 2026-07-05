#!/usr/bin/env python3
"""
音频播放模块 - 强制双声道输出
"""
import subprocess
import os

class AudioPlayer:
    def __init__(self, card=0, device=0, volume=200):
        self.card = card
        self.device = device
        self.volume = volume
        
        self.temp_mp3 = '/tmp/tts_temp.mp3'
        self.temp_wav = '/tmp/tts_temp.wav'
        
        self._set_volume(volume)
        
        try:
            import edge_tts
            self.edge_tts_available = True
            print('✅ edge-tts 已加载')
        except ImportError:
            self.edge_tts_available = False
            print('⚠️ edge-tts 不可用')
    
    def _set_volume(self, volume):
        try:
            subprocess.run(f'tinymix -D {self.card} set 7 {volume}', 
                         shell=True, capture_output=True)
        except:
            pass
    
    def _play_wav(self, wav_file):
        if not os.path.exists(wav_file):
            return False
        
        try:
            # 用 plughw 播放（允许多进程共享声卡，避免 PyAudio 占用时 aplay 声音异常）
            cmd = f'aplay -D plughw:{self.card},{self.device} {wav_file} 2>/dev/null'
            subprocess.run(cmd, shell=True, check=True)
            return True
        except:
            return False
    
    def say(self, text):
        if not self.edge_tts_available:
            print(f'🔊 [语音] {text}')
            return False
        
        try:
            import edge_tts
            
            # 合成 MP3
            tts = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")
            tts.save_sync(self.temp_mp3)
            
            # 转换为 WAV，强制双声道、48000Hz
            cmd = f'ffmpeg -y -i {self.temp_mp3} -ac 2 -ar 48000 {self.temp_wav} 2>/dev/null'
            subprocess.run(cmd, shell=True, check=True)
            
            # 播放
            result = self._play_wav(self.temp_wav)
            return result
            
        except Exception as e:
            print(f'❌ TTS 失败: {e}')
            print(f'🔊 [语音] {text}')
            return False
    
    def say_hello(self, name):
        self.say(f'你好，{name}')
    
    def say_welcome(self, name):
        self.say(f'欢迎你，{name}')

if __name__ == '__main__':
    player = AudioPlayer(volume=200)
    player.say('你好，我是机器人')
