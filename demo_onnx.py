#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from pathlib import Path
from funasr_onnx import SenseVoiceSmall
#from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
import time

model_dir = "iic/SenseVoiceSmall"

model = SenseVoiceSmall(model_dir, batch_size=10, quantize=True)

# inference
#wav_or_scp = ["{}/.cache/modelscope/hub/models/{}/example/en.mp3".format(Path.home(), model_dir)]
wav_or_scp = ["/Users/sunxuguang/SenseVoice/通话记录-598320-user-转人工.wav"]
time1 = time.perf_counter()
res = model(wav_or_scp, language="auto", textnorm="withitn")
time2 = time.perf_counter()
print(res)
print(f"cost time: {time2 - time1:0.3f}")
#print([rich_transcription_postprocess(i) for i in res])
