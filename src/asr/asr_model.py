import speech_recognition as sr
import os
import contextlib
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tempfile
import wave
import pyaudio
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

class ASRModule:
    def __init__(self, use_sensevoice=True, model_dir="./model/SenseVoice"):
        # 初始化语音识别器
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.use_sensevoice = use_sensevoice
        self.sensevoice_model = None
        
        # 配置识别器参数
        self.recognizer.energy_threshold = 300  # 麦克风灵敏度
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 1.0    # 语音暂停时间
        
        # 初始化SenseVoice模型
        if use_sensevoice:
            self._init_sensevoice(model_dir)
    
    def calibrate_noise_level(self, duration=1.5):
        """
        专门的环境噪音校准方法（已禁用）
        :param duration: 校准时长（秒）
        """
        # 已取消环境噪音校准
        pass
        
    def _init_sensevoice(self, model_dir):
        """
        初始化SenseVoice模型
        """
        try:
            print("正在加载SenseVoice模型...")
            self.sensevoice_model = AutoModel(
                model=model_dir,
                trust_remote_code=True,
                device="cpu",  # 使用CPU推理
            )
            print("SenseVoice模型加载完成")
        except Exception as e:
            print(f"加载SenseVoice模型失败: {e}")
            self.use_sensevoice = False
            print("将使用默认的Google语音识别")
    
    def record_audio(self, duration=5):
        """
        使用麦克风录制音频
        :param duration: 录制时长（秒）
        :return: 录制的音频数据
        """
        print("正在录音...")
        with self.microphone as source:
            # 已取消环境噪音校准
            audio = self.recognizer.listen(source, timeout=duration)
        print("录音结束")
        return audio
    
    def record_audio_to_wav(self, duration=5, filename="temp_recording.wav"):
        """
        使用PyAudio录制音频到WAV文件
        :param duration: 录制时长（秒）
        :param filename: 输出文件名
        :return: 录制的音频文件路径
        """
        FORMAT = pyaudio.paInt16  # 16位PCM格式，提供更好的音质
        CHANNELS = 1              # 单声道，符合SenseVoice要求
        RATE = 16000              # 16kHz采样率，SenseVoice需要
        CHUNK = 2048              # 增加缓冲区大小，提高录音质量和性能
        
        audio = pyaudio.PyAudio()
        
        # 开始录制
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                           input=True, frames_per_buffer=CHUNK)
        
        print(f"正在录音 {duration} 秒...")
        
        frames = []
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        print("录音结束")
        
        # 停止录制
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # 保存到WAV文件
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return filename
        
    def record_audio_vad(self, filename="temp_recording.wav", vad_level=3, max_silence_duration=2.0, sample_rate=16000):
        """
        使用VAD（Voice Activity Detection）进行不定长录音
        :param filename: 输出文件名
        :param vad_level: VAD灵敏度（0-3，3最敏感）
        :param max_silence_duration: 最大静音时长（秒），超过此时间则停止录音
        :param sample_rate: 采样率
        :return: 录制的音频文件路径
        """
        import webrtcvad
        
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        CHUNK_DURATION_MS = 30  # 30ms块
        CHUNK_SIZE = int(sample_rate * CHUNK_DURATION_MS / 1000)  # 每块的样本数
        
        # 初始化VAD
        vad = webrtcvad.Vad(vad_level)
        
        # 初始化PyAudio
        audio = pyaudio.PyAudio()
        
        # 打开音频流
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=sample_rate,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        
        print("开始不定长录音（检测到语音停顿将自动停止）...")
        
        frames = []
        voice_activity = []
        silent_chunks = 0
        max_silent_chunks = int(max_silence_duration * 1000 / CHUNK_DURATION_MS)
        recording_started = False
        
        try:
            while True:
                chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                
                # VAD检测
                is_speech = vad.is_speech(chunk, sample_rate)
                voice_activity.append(is_speech)
                
                # 只保留最近50个块的信息
                if len(voice_activity) > 50:
                    voice_activity.pop(0)
                
                # 检查是否有语音活动
                has_voice = any(voice_activity)
                
                if has_voice and not recording_started:
                    # 检测到语音，开始录制
                    recording_started = True
                    print("检测到语音，开始录音...")
                
                if recording_started:
                    frames.append(chunk)
                    
                    if is_speech:
                        # 检测到语音，重置静音计数器
                        silent_chunks = 0
                    else:
                        # 检测到静音，增加静音计数器
                        silent_chunks += 1
                        
                        # 检查是否超过最大静音时长
                        if silent_chunks > max_silent_chunks:
                            print("检测到语音停顿，停止录音")
                            break
        except KeyboardInterrupt:
            print("\n用户中断录音")
        finally:
            # 停止录音
            stream.stop_stream()
            stream.close()
            audio.terminate()
        
        if not recording_started:
            print("未检测到有效语音")
            return None
        
        # 保存到WAV文件
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return filename
    
    def recognize_speech(self, audio, max_attempts=3, confidence_threshold=60):
        """
        将音频转换为文本，添加置信度评估和重复识别机制
        :param audio: 音频数据或文件路径
        :param max_attempts: 最大尝试次数
        :param confidence_threshold: 置信度阈值（低于此值将进行重复识别）
        :return: 识别结果文本
        """
        best_result = ""
        best_confidence = 0
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            # 优先尝试SenseVoice识别
            if self.use_sensevoice and self.sensevoice_model:
                result = self._recognize_sensevoice(audio)
                # 如果SenseVoice识别失败，回退到Google语音识别
                if result.strip() == "":
                    result = self._recognize_google(audio)
            else:
                # 直接使用Google语音识别
                result = self._recognize_google(audio)
            
            # 应用后处理优化
            processed_result = self.postprocess_result(result)
            
            # 评估置信度
            confidence = self.evaluate_confidence(processed_result)
            
            # 更新最佳结果
            if confidence > best_confidence:
                best_confidence = confidence
                best_result = processed_result
                
            # 如果置信度足够高，提前返回
            if confidence >= confidence_threshold:
                break
        
        # 只输出最终结果
        return best_result
    
    def _recognize_google(self, audio):
        """
        使用Google语音识别
        :param audio: AudioData对象或文件路径
        :return: 识别结果文本
        """
        try:
            # 如果是文件路径，先读取为AudioData对象
            if not isinstance(audio, sr.AudioData):
                with sr.AudioFile(audio) as source:
                    audio_data = self.recognizer.record(source)
            else:
                audio_data = audio
            
            text = self.recognizer.recognize_google(audio_data, language='zh-CN')
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""
        except Exception:
            return ""
    
    def _recognize_sensevoice(self, audio):
        """
        使用SenseVoice识别语音
        """
        try:
            audio_path = ""
            
            # 使用ExitStack管理临时文件，确保无论发生什么都能正确清理
            with contextlib.ExitStack() as stack:
                # 如果是SpeechRecognition的AudioData对象，先保存为WAV文件
                if isinstance(audio, sr.AudioData):
                    # 创建临时WAV文件
                    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    stack.callback(lambda: os.unlink(temp_wav.name) if os.path.exists(temp_wav.name) else None)
                    temp_wav.write(audio.get_wav_data())
                    temp_wav.close()
                    audio_path = os.path.abspath(temp_wav.name)
                    
                else:
                    # 如果是文件路径，转换为绝对路径
                    original_audio_path = os.path.abspath(audio)
                    
                    # 检查文件格式，如果不是WAV，转换为WAV
                    if not original_audio_path.lower().endswith('.wav'):
                        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        stack.callback(lambda: os.unlink(temp_wav.name) if os.path.exists(temp_wav.name) else None)
                        temp_wav.close()
                        
                        try:
                            # 使用pydub转换音频格式
                            sound = AudioSegment.from_file(original_audio_path)
                            sound = sound.set_frame_rate(16000)  # 设置为16kHz
                            sound = sound.set_channels(1)       # 设置为单声道
                            sound.export(temp_wav.name, format="wav")
                            audio_path = temp_wav.name
                        except Exception as e:
                            return ""
                    else:
                        # 直接使用原始文件路径
                        audio_path = original_audio_path
            
            # 确保文件存在
            if not os.path.exists(audio_path):
                print(f"  错误: 文件不存在 {audio_path}")
                return ""
            
            # 使用SenseVoice识别，优化参数以提高准确率和性能
            res = self.sensevoice_model.generate(
                input=audio_path,
                cache={},
                language="zh",     # 直接指定中文，提高识别速度和准确率
                use_itn=True,      # 使用逆文本正则化
                beam_size=5,       # 增加beam搜索宽度，提高识别准确率
                batch_size=1,      # 批处理大小，根据硬件性能调整
            )
            
            # 处理识别结果
            text = rich_transcription_postprocess(res[0]["text"])
            return text
        except Exception:
            return ""
    
    def postprocess_result(self, text):
        """
        识别结果的后处理优化
        :param text: 原始识别结果
        :return: 优化后的识别结果
        """
        if not text:
            return ""
        
        # 1. 去除首尾空格和换行符
        text = text.strip()
        
        # 2. 过滤无效字符和标点符号
        import re
        # 保留中文、英文、数字和常见标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：、,.!?;:\s]', '', text)
        
        # 3. 修正常见的识别错误
        common_errors = {
            '维': '我',
            '为': '我',
            '末': '没',
            '么': '没',
            '那': '那',
            '哪': '哪',
            '的': '的',
            '地': '地',
            '得': '得',
        }
        
        for error, correct in common_errors.items():
            text = text.replace(error, correct)
        
        # 4. 去除重复的字符（连续出现3次以上的）
        text = re.sub(r'(.)\1{3,}', r'\1', text)
        
        # 5. 去除多余的空格
        text = re.sub(r'\s+', ' ', text)
        
        return text
        
    def evaluate_confidence(self, text):
        """
        评估识别结果的置信度
        :param text: 识别结果文本
        :return: 置信度分数（0-100）
        """
        if not text:
            return 0
            
        import re
        confidence = 100
        
        # 1. 长度检查
        if len(text) < 2:
            confidence -= 40
        elif len(text) < 5:
            confidence -= 20
        
        # 2. 检查是否包含无效或不连贯内容
        if re.search(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：、,.!?;:\s]', text):
            confidence -= 30
            
        # 3. 检查是否只有标点符号或特殊字符
        if re.match(r'^[，。！？；：、,.!?;:\s]+$', text):
            confidence -= 50
            
        # 4. 检查是否包含大量重复字符
        if re.search(r'(.)\1{2,}', text):
            confidence -= 20
            
        # 5. 检查是否包含常见的识别错误模式
        error_patterns = ['嗯嗯', '啊啊', '哦哦', '呃呃', '哼哼']
        for pattern in error_patterns:
            if pattern in text:
                confidence -= 15
                break
                
        # 确保置信度在0-100之间
        return max(0, min(100, confidence))
        
    def record_and_recognize(self, duration=5, use_vad=True, **vad_kwargs):
        """
        录制并识别语音
        :param duration: 录制时长（秒，仅当use_vad=False时有效）
        :param use_vad: 是否使用VAD不定长录音
        :param vad_kwargs: VAD相关参数
        :return: 识别结果文本
        """
        # 在录制前进行环境噪音校准（已禁用）
        self.calibrate_noise_level()
        
        if self.use_sensevoice and self.sensevoice_model:
            # 使用SenseVoice时录制为WAV文件
            if use_vad:
                # 使用VAD不定长录音
                wav_file = self.record_audio_vad(**vad_kwargs)
            else:
                # 使用固定时长录音
                wav_file = self.record_audio_to_wav(duration)
            
            if wav_file:
                return self.recognize_speech(wav_file)
            else:
                return ""
        else:
            # 使用Google时直接使用AudioData（不支持VAD）
            audio = self.record_audio(duration)
            if audio:
                return self.recognize_speech(audio)
            return ""

if __name__ == "__main__":
    # 测试ASR模块
    asr = ASRModule(use_sensevoice=True)
    print("请说话...")
    text = asr.record_and_recognize(duration=5)
    if text:
        print(f"最终识别结果: {text}")
    else:
        print("未识别到有效语音")