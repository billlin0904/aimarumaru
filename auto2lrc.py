from separate import AudioSeparate
from faster_whisper import WhisperModel
import gc
import os

class Auto2Lrc:
    def __init__(self, model_name="large-v2", device="cuda", compute_type="float16"):
        """
        初始化 faster_whisper 模型
        :param model_name: Whisper 模型名稱 (如 'large-v3')
        :param device: 選擇設備 (如 'cuda' 或 'cpu')
        :param compute_type: 計算類型，'float16' 或 'int8' (低 GPU 記憶體可選擇 int8)
        """
        self.model = WhisperModel(model_size_or_path = model_name, device = device, compute_type = compute_type)
        self.separate = AudioSeparate(model_path="uvr5_weights/2_HP-UVR.pth", device=device)
        
    def separate_audio(self, file_path: str):        
        ins_root = "vocal"
        name = os.path.basename(file_path)
        self.separate.save_audio(file_path, None, vocal_root=ins_root)
        save_path = os.path.join(ins_root, f'vocal_{name}.wav')
        return save_path
        
    def save_as_lrc(self, segments, output_file):
        """
        將轉錄段落保存為 LRC 格式文件，適配 faster_whisper 的輸出格式
        :param segments: faster_whisper 模型轉錄後的段落結果 (列表，包含 'start', 'end', 'text' 字段)
        :param output_file: LRC 文件輸出路徑
        """
        def format_time(seconds):
            """將時間轉換為 LRC 格式的時間戳 [mm:ss.xx]"""
            mins, secs = divmod(seconds, 60)
            return f"[{int(mins):02}:{secs:05.2f}]"
    
        with open(output_file, 'w', encoding='utf-8') as f:
            for segment in segments:
                # 獲取開始時間和文本
                start_time = format_time(segment.start)  # faster_whisper 返回的是對象字段
                text = segment.text.strip()  # 提取轉錄後的文本，並去除前後空格
                f.write(f"{start_time} {text}\n")  # 按 LRC 格式寫入時間和文本
    
        print(f"LRC 文件已保存至: {output_file}")

    
    def get_lrc(self, audio_file, output_lrc_file):
        save_path = self.separate_audio(audio_file)
         
        try:       
           segments, info = self.model.transcribe(save_path, beam_size=5)
           # 保存為 LRC 文件
           self.save_as_lrc(segments, output_lrc_file)           
        finally:
            os.remove(save_path)
            self.clear_model_cache()

    def clear_model_cache(self):
        """
        清除 GPU 記憶體中的模型以節省資源
        """
        gc.collect()
        import torch
        torch.cuda.empty_cache()


