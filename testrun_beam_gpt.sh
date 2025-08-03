export OMP_NUM_THREADS=1
export PROJECT=/ocean/projects/cis250053p/dsreedha

python -m emg2qwerty.train \
  user=user0,user1,user2,user3 \
  checkpoint=${PROJECT}/emg2qwerty/models/personalized-conformer/user0.ckpt,${PROJECT}/emg2qwerty/models/personalized-conformer/user1.ckpt,${PROJECT}/emg2qwerty/models/personalized-conformer/user2.ckpt,${PROJECT}/emg2qwerty/models/personalized-conformer/user3.ckpt \
  train=False trainer.accelerator=cpu \
  decoder=ctc_greedy_gpt \
  hydra.launcher.mem_gb=64 \
  --multirun
