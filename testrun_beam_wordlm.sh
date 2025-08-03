#!/bin/bash
export PROJECT=/ocean/projects/cis250053p/dsreedha
export PYTHONPATH=$PYTHONPATH:$PROJECT

python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${PROJECT}/emg2qwerty/models/personalized-conformer/\${user}.ckpt" \
  train=False \
  trainer.accelerator=gpu \
  decoder=ctc_beam_word_lm \
  decoder.verbose=true decoder.debug=false \
  hydra.launcher.mem_gb=64 \
  --multirun
