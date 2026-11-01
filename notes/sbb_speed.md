# SBB speed (laptop: i7-13650HX, 14 cores / 20 threads, 31 GB)

Measured with copies of `scripts/sbb2022_masks.py` writing to a scratch folder; masks compared pixel by pixel with
`data/crops/sbb2022_raw/`.

- **SBB is nearly all the time.** 30 letters: 39.5 of 48.8 s in SBB; loading the model 6.5 s once per copy; opening a
  page 0.1 s. SBB runs a 512 x 512 region per letter as four 448 x 448 tiles.
- **One copy: 0.91 to 1.32 s per letter** (laptop otherwise quiet, or with another job on the GPU and one core).
- **`--workers 4` is the speed-up kept: 120 letters in 79 s, against 123 s for one copy.** 6 copies 73 s, 8 copies
  84 s; 4 copies measured 107 s in an earlier run, so runs vary by about a third. 0 differing pixels in all 120 letters
  for 1, 4, 6 and 8 copies. About 1.8 GB of memory per copy.
- **Not faster.** Calling the network directly instead of Keras `predict()`: 58 s against 39 s for 30 letters. Four
  copies capped at 4 TensorFlow threads each: 121 s against 107 s uncapped.
- **SBB once per page changes the masks.** A page needs about 90 tiles, so at 45 or more letters a page it does about
  half the work, but 1.9% of box pixels (7.7% of the ink) differ, in all 30 letters tested: each letter sees different
  surroundings in its tiles.
- **The GPU is faster still: 120 letters, SBB 41 s (0.34 s per letter), 54.6 s in all, 0 differing pixels.** Shared
  with a training job on the GPU: 0.72 s per letter. Environment `voynich-sbb-gpu` (`requirements-sbb-gpu.txt`):
  TensorFlow 2.21 with tf_keras, run with `TF_USE_LEGACY_KERAS=1`, TF32 turned off
  (`tf.config.experimental.enable_tensor_float_32_execution(False)`), and `LD_LIBRARY_PATH` listing every
  `site-packages/nvidia/*/lib` folder; without it cuSOLVER fails to load and TensorFlow quietly uses the CPU. The first
  run compiles GPU code for the RTX 5060 (keep it with `CUDA_CACHE_MAXSIZE=4294967296`). TensorFlow 2.21 on the CPU is
  far slower (13 letters in 10 min): use that environment on the GPU only.
- **Not tested:** Sonic.
- **Other steps, per letter on the CPU:** U-Net reading and inference 28 ms (`scripts/k_segmenter.py`); DINOv3
  0.1 s on 8 threads (`scripts/embed.py`). U-Net training is a fixed size (6,000 composites a fold): about 162 s a fold
  on the GPU in the last run, from the output files' times.
