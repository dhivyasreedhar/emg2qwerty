export PROJECT=/ocean/projects/cis250053p/dsreedha
python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${PROJECT}/emg2qwerty/models/personalized-conformer/\${user}.ckpt"\
  train=False trainer.accelerator=cpu \
  decoder=ctc_beam_flan\
  hydra.launcher.mem_gb=64 \
  --multirun
