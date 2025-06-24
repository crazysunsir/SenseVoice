#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from model import SenseVoiceSmall
import time
#from funasr.utils.postprocess_utils import rich_transcription_postprocess


model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cpu")
m.eval()

# all results
time1 = time.perf_counter()
res = m.inference(
    data_in=f"/Users/sunxuguang/SenseVoice/通话记录-598320-user-转人工.wav",
    language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    ban_emo_unk=False,
    **kwargs,
)
time2 = time.perf_counter()
print(res)
print(f"cost time: {time2 - time1:0.3f}")

#only emo results
time1 = time.perf_counter()
emo = m.inference_emo(
    data_in=f"/Users/sunxuguang/SenseVoice/通话记录-598320-user-转人工.wav",
    language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    ban_emo_unk=False,
    **kwargs,
)
time2 = time.perf_counter()
print(emo)
print(f"cost time: {time2 - time1:0.3f}")