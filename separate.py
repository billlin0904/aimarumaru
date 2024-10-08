import os, torch, warnings
warnings.filterwarnings("ignore")

import librosa
import importlib
import  numpy as np
import hashlib , math

from uvr5_pack.lib_v5 import spec_utils
from uvr5_pack.utils import _get_name_params,inference
from uvr5_pack.lib_v5.model_param_init import ModelParameters

from scipy.io import wavfile

class AudioSeparate:
    def __init__(self, model_path: str, device: str = "cuda", is_half: bool = True):
        """
        初始化 AudioSeparate 類別，並加載指定的模型檔案。
        
        參數:
        ----------
        model_path : str
            用於音頻分離的預訓練模型的文件路徑。
        
        device : str, 預設為 'cuda'
            用於運行模型的設備，'cuda' 表示使用 GPU 加速運算。也可以設置為 'cpu'。
        
        is_half : bool, 預設為 True
            是否使用半精度浮點數運行模型。啟用半精度可以減少運算時間和內存使用，但在某些情況下可能影響精度。
        """
        self.model_path = model_path
        self.device = device
        self.data = {
            'postprocess': False,  # 後處理選項，控制是否在分離後進行額外處理
            'tta': False,  # 測試時增強（Test-Time Augmentation）選項
            'window_size': 512,  # 頻譜分析窗口大小
            'agg': 10,  # 激進程度，影響人聲和伴奏的分離強度
            'high_end_process': 'mirroring',  # 高頻處理方式，默認使用鏡像技術
        }
        
        # 支持的神經網絡模型大小
        nn_arch_sizes = [
            31191,  # 默認大小
            33966, 61968, 123821, 123812, 537238  # 自定義大小
        ]
        
        # 根據模型文件大小選擇合適的神經網絡架構
        self.nn_architecture = list('{}KB'.format(s) for s in nn_arch_sizes)
        model_size = math.ceil(os.stat(model_path).st_size / 1024)
        nn_architecture = '{}KB'.format(min(nn_arch_sizes, key=lambda x: abs(x - model_size)))
        
        # 動態導入相應的神經網絡架構
        nets = importlib.import_module('uvr5_pack.lib_v5.nets' + f'_{nn_architecture}'.replace('_{}KB'.format(nn_arch_sizes[0]), ''), package=None)
        
        # 計算模型檔案的哈希值，用於確認模型的參數
        model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()
        
        # 加載模型參數
        param_name, model_params_d = _get_name_params(model_path, model_hash)
        self.mp = ModelParameters(model_params_d)
        
        # 初始化模型，並加載預訓練權重
        model = nets.CascadedASPPNet(self.mp.param['bins'] * 2)
        cpk = torch.load(model_path, map_location='cpu')
        model.load_state_dict(cpk)
        model.eval()  # 設置模型為推理模式
        
        # 決定是否使用半精度浮點數
        if is_half:
            model = model.half().to(device)
        else:
            model = model.to(device)

        self.model = model
        
    def save_audio(self, music_file, ins_root=None, vocal_root=None):
        """
        將輸入的音樂文件進行人聲和樂器聲的分離，並保存為單獨的音頻文件。
        
        參數:
        ----------
        music_file : str
            要處理的音樂文件的路徑。該文件將被加載並進行人聲和伴奏的分離處理。
        
        ins_root : Optional[str], 預設為 None
            分離出的樂器聲（伴奏）保存的目錄路徑。如果提供了該路徑，伴奏部分將保存為 WAV 文件。
            若該路徑為 None，則不保存樂器聲文件。
        
        vocal_root : Optional[str], 預設為 None
            分離出的人聲保存的目錄路徑。如果提供了該路徑，人聲部分將保存為 WAV 文件。
            若該路徑為 None，則不保存人聲文件。
        
        返回值:
        ----------
        str:
            如果 `ins_root` 和 `vocal_root` 均為 None，則返回 "No save root."，表示沒有指定保存路徑。
            否則，不返回任何值，直接將文件保存到指定路徑中。
        """
    
        # 檢查是否提供了存儲路徑，如果都為 None 則返回提示
        if(ins_root is None and vocal_root is None):
            return "No save root."
    
        # 獲取音樂文件的文件名
        name = os.path.basename(music_file)
        
        # 如果提供了 ins_root 或 vocal_root，則創建相應的目錄
        if(ins_root is not None):
            os.makedirs(ins_root, exist_ok=True)
        if(vocal_root is not None):
            os.makedirs(vocal_root, exist_ok=True)
        
        # 初始化字典來存儲波形和頻譜數據
        X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
        # 取得處理所需的頻帶數量
        bands_n = len(self.mp.param['band'])
        
        # 逐步處理每個頻帶，從高頻到低頻
        for d in range(bands_n, 0, -1):
            bp = self.mp.param['band'][d]
            # 如果是最高頻帶，則直接從文件中加載波形數據
            if d == bands_n:
                X_wave[d], _ = librosa.core.load(
                    music_file, sr = bp['sr'], mono = False, dtype=np.float32, res_type=bp['res_type'])
                # 如果音頻數據是一維的，則將其轉換為兩個通道的格式
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
            else:
                # 如果是較低頻帶，則進行重採樣
                X_wave[d] = librosa.core.resample(X_wave[d+1], orig_sr = self.mp.param['band'][d+1]['sr'], target_sr = bp['sr'], res_type=bp['res_type'])
            
            # 將波形數據轉換為頻譜
            X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], 
                                                        self.mp.param['mid_side'], 
                                                        self.mp.param['mid_side_b2'], 
                                                        self.mp.param['reverse'])
            
            # 處理高頻數據
            if d == bands_n and self.data['high_end_process'] != 'none':
                input_high_end_h = (bp['n_fft'] // 2 - bp['crop_stop']) + (self.mp.param['pre_filter_stop'] - self.mp.param['pre_filter_start'])
                input_high_end = X_spec_s[d][:, bp['n_fft'] // 2 - input_high_end_h:bp['n_fft'] // 2, :]
    
        # 將所有頻帶的頻譜合併
        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        
        # 設定分離的激進程度
        aggresive_set = float(self.data['agg'] / 100)
        aggressiveness = {'value': aggresive_set, 'split_bin': self.mp.param['band'][1]['crop_stop']}
        
        # 使用模型進行推理，分離人聲和樂器聲
        with torch.no_grad():
            pred, X_mag, X_phase = inference(X_spec_m, self.device, self.model, aggressiveness, self.data)
        
        # 後處理步驟，根據需要進行消除噪音處理
        if self.data['postprocess']:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
        
        # 生成人聲和樂器的頻譜
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m
    
        # 處理和保存樂器聲音頻
        if ins_root is not None:
            if self.data['high_end_process'].startswith('mirroring'):
                input_high_end_ = spec_utils.mirroring(self.data['high_end_process'], y_spec_m, input_high_end, self.mp)
                wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp, input_high_end_h, input_high_end_)
            else:
                wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)
            print('%s instruments done' % name)
            wavfile.write(os.path.join(ins_root, 'instrument_{}.wav'.format(name)), 
                          self.mp.param['sr'], (np.array(wav_instrument) * 32768).astype("int16"))
    
        # 處理和保存人聲音頻
        if vocal_root is not None:
            if self.data['high_end_process'].startswith('mirroring'):
                input_high_end_ = spec_utils.mirroring(self.data['high_end_process'], v_spec_m, input_high_end, self.mp)
                wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp, input_high_end_h, input_high_end_)
            else:
                wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)
            print('%s vocals done' % name)
            wavfile.write(os.path.join(vocal_root, 'vocal_{}.wav'.format(name)), 
                          self.mp.param['sr'], (np.array(wav_vocals) * 32768).astype("int16"))
