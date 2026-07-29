import gc
import logging
import os

from text_converter import to_traditional_chinese


logger = logging.getLogger(__name__)


class Auto2Lrc:
    def __init__(self, model_name="large-v3", device="cuda", compute_type="float16", separator_model_path="uvr5_weights/2_HP-UVR.pth"):
        """
        初始化 faster_whisper 模型
        :param model_name: Whisper 模型名稱 (如 'large-v3')
        :param device: 選擇設備 (如 'cuda' 或 'cpu')
        :param compute_type: 計算類型，'float16' 或 'int8' (低 GPU 記憶體可選擇 int8)
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.separator_model_path = separator_model_path
        self.separate = None

    def get_model(self):
        if self.model is None:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(model_size_or_path = self.model_name, device = self.device, compute_type = self.compute_type)
        return self.model

    def get_separator(self):
        if self.separate is None:
            from separate import AudioSeparate

            self.separate = AudioSeparate(model_path=self.separator_model_path, device=self.device)
        return self.separate
        
    def separate_audio(self, file_path: str):        
        ins_root = "vocal"
        name = os.path.basename(file_path)
        self.get_separator().save_audio(file_path, None, vocal_root=ins_root)
        save_path = os.path.join(ins_root, f'vocal_{name}.wav')
        return save_path
        
    def save_as_lrc(self, segments, output_file, language=None):
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
                text = to_traditional_chinese(segment.text.strip(), language)
                f.write(f"{start_time} {text}\n")  # 按 LRC 格式寫入時間和文本
    
        logger.info("LRC 文件已保存至: %s", output_file)

    def save_as_srt(self, segments, output_file, language=None):
        """
        將轉錄段落保存為 SRT 字幕格式。
        """
        def format_time(seconds):
            millis = int(round(seconds * 1000))
            hours, remainder = divmod(millis, 3600000)
            minutes, remainder = divmod(remainder, 60000)
            secs, millis = divmod(remainder, 1000)
            return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

        with open(output_file, 'w', encoding='utf-8') as f:
            for index, segment in enumerate(segments, start=1):
                start_time = format_time(segment.start)
                end_time = format_time(segment.end)
                text = to_traditional_chinese(segment.text.strip(), language)
                f.write(f"{index}\n{start_time} --> {end_time}\n{text}\n\n")

        logger.info("SRT 文件已保存至: %s", output_file)

    def save_as_text(self, segments, output_file, language=None):
        """
        將轉錄段落保存為純文字格式。
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for segment in segments:
                text = to_traditional_chinese(segment.text.strip(), language)
                if text:
                    f.write(f"{text}\n")

        logger.info("純文字文件已保存至: %s", output_file)

    def get_srt(self, audio_file, output_srt_file, language=None, beam_size=5):
        segments, info = self.get_model().transcribe(audio_file, beam_size=beam_size, language=language)
        self.save_as_srt(segments, output_srt_file, getattr(info, "language", language))
        self.clear_model_cache()
        return info

    def get_text(self, audio_file, output_text_file, language=None, beam_size=5):
        segments, info = self.get_model().transcribe(audio_file, beam_size=beam_size, language=language)
        self.save_as_text(segments, output_text_file, getattr(info, "language", language))
        self.clear_model_cache()
        return info

    
    def get_lrc(self, audio_file, output_lrc_file, use_vocal_separation=False):
        source_path = audio_file
        separated_path = None

        if use_vocal_separation and self.separator_model_path and os.path.exists(self.separator_model_path):
            separated_path = self.separate_audio(audio_file)
            source_path = separated_path

        try:
           segments, info = self.get_model().transcribe(source_path, beam_size=5)
           # 保存為 LRC 文件
           self.save_as_lrc(segments, output_lrc_file, getattr(info, "language", None))
        finally:
            if separated_path and os.path.exists(separated_path):
                os.remove(separated_path)
            self.clear_model_cache()

    def clear_model_cache(self):
        """
        清除 GPU 記憶體中的模型以節省資源
        """
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
