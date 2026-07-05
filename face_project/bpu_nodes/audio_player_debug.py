#!/usr/bin/env python3
"""
音频播放模块 - 带调试信息
"""
import subprocess
import os
import tempfile

class AudioPlayer:
    def __init__(self, card=0, device=0, volume=200):
        self.card = card
        self.device = device
        self.volume = volume
        
        self.temp_mp3 = '/tmp/tts_temp.mp3'
        self.temp_wav = '/tmp/tts_temp.wav'
        
        print(f'🔧 初始化: card={card}, device={device}, volume={volume}')
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
            result = subprocess.run(f'tinymix -D {self.card} set 7 {volume}', 
                         shell=True, capture_output=True, text=True)
            print(f'🔊 设置音量: {volume}, 输出: {result.stdout}')
        except Exception as e:
            print(f'❌ 设置音量失败: {e}')
    
    def _play_wav(self, wav_file):
        print(f'🔍 播放文件: {wav_file}')
        if not os.path.exists(wav_file):
            print(f'❌ 文件不存在: {wav_file}')
            return False
        
        file_size = os.path.getsize(wav_file)
        print(f'📁 文件大小: {file_size} 字节')
        
        try:
            cmd = f'aplay -D hw:{self.card},{self.device} {wav_file}'
            print(f'🔊 执行: {cmd}')
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(f'📤 返回码: {result.returncode}')
            if result.stdout:
                print(f'📤 stdout: {result.stdout}')
            if result.stderr:
                print(f'📤 stderr: {result.stderr}')
            return result.returncode == 0
        except Exception as e:
            print(f'❌ 播放异常: {e}')
            return False
    
    def say(self, text):
        print(f'🔊 合成语音: "{text}"')
        
        if not self.edge_tts_available:
            print(f'⚠️ edge-tts 不可用，只打印文字')
            return False
        
        try:
            import edge_tts
            
            print(f'🔄 合成中...')
            tts = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")
            tts.save_sync(self.temp_mp3)
            print(f'✅ MP3 已保存: {self.temp_mp3}')
            
            print(f'🔄 转换 MP3 -> WAV...')
            cmd = f'ffmpeg -y -i {self.temp_mp3} {self.temp_wav}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f'❌ ffmpeg 失败: {result.stderr}')
                return False
            print(f'✅ WAV 已保存: {self.temp_wav}')
            
            print(f'🔄 播放中...')
            result = self._play_wav(self.temp_wav)
            print(f'✅ 播放完成: {result}')
            return result
            
        except Exception as e:
            print(f'❌ TTS 失败: {e}')
            print(f'🔊 [语音] {text}')
            return False
    
    def say_hello(self, name):
        self.say(f'你好，{name}')

if __name__ == '__main__':
    player = AudioPlayer(volume=200)
    player.say('你好，陈伟凡')
