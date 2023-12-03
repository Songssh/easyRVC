from scipy.io import wavfile
from fairseq import checkpoint_utils
from whisper.audio import load_audio
from rvc_infer.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from modules.vc_infer_pipeline import VC
from multiprocessing import cpu_count
import numpy as np
import torch
from modules.hparams import hparams

class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available() and self.device != "cpu":
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(f"configs/{config_file}", "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(f"configs/{config_file}", "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            self.is_half = False

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


class Rvc:
    def __init__(self, model, f0_method):
        #self.device = "cuda:0"
        self.device = hparams['device']
        self.is_half = False
        self.config = Config(self.device, self.is_half)
        self.hubert_model = self.load_hubert()
        
        self.sid = 0
        #self.f0_up_key = 0
        #self.f0_file = None
        #self.f0_method = f0_method
        self.file_index=""
        self.file_index2=""
        #self.index_rate=0.75
        #self.filter_radius=3
        #self.resample_sr=0
        #self.rms_mix_rate=0.25
        #self.model_path=model
        #self.input_audio_path = "testdata/up2.wav"
        #self.output_path="outputs/test.wav"
        
        #self.cpt, self.tgt_sr, self.version,\
        #self.net_g, self.vc, self.n_spk = self.load_vc(model)

    def load_vc(self, model_path):
        print("loading pth %s" % model_path)
        cpt = torch.load(model_path, map_location="cpu")
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")
        if version == "v1":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=self.is_half)
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        elif version == "v2":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=self.is_half)
            else:
                net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
        del net_g.enc_q
        print(net_g.load_state_dict(cpt["weight"], strict=False))
        net_g.eval().to(self.device)
        net_g = net_g.float()
        vc = VC(tgt_sr, self.config)
        n_spk = cpt["config"][-3]
        return cpt, tgt_sr, version, net_g, vc, n_spk

    def load_hubert(self):
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [hparams['hubert_path']],
            suffix="",
        )
        hubert_model = models[0]
        hubert_model = hubert_model.to(self.device)
        hubert_model = hubert_model.float()
        hubert_model.eval()
        return hubert_model

    def convert(self,
                input_audio_path,
                output_path,
                model_path,
                f0_up_key,
                f0_file,
                f0_method,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                ):

        cpt, tgt_sr, version,\
        net_g, vc, n_spk = self.load_vc(model_path)
        
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        times = [0, 0, 0]

        if audio_max > 1:
            audio /= audio_max
            
        if_f0 = cpt.get("f0", 1)
        audio_opt = vc.pipeline(
            self.hubert_model,
            net_g,
            self.sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            self.file_index,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            f0_file=f0_file,
            protect=0.33,
        )
        wavfile.write(output_path, tgt_sr, audio_opt)
        return f"Success: {output_path}", (tgt_sr, audio_opt)

def test():
    rvc = Rvc(hparams['rvc_checkpoint'], hparams['f0_method'])
    rvc.convert("test.mp3",
                "tt.wav",
                "data/models/rvc/test100.pth",
                0,
                None,
                "rmvpe",
                0.75,
                3,
                0,
                0.25,
                )

if __name__ == "__main__":
    test()
