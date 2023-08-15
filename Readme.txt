1) Create environment matching environment.yaml
2) Clone stylegan3 from https://github.com/NVlabs/stylegan3/tree/main into the folder "stylegan3"
3) Clone WarpedGANSpace from https://github.com/chi0tzp/WarpedGANSpace into the folder "WarpedGANSpace"
3) a) In lib/__init__ and lib/trainer rename imports from .aux to ._aux to link to the correct file
3) b) Run python download_models.py to download pretrained models


What to set for GPUs:

datagen.py
	prepare_ffdnet

trainer.py
	train_own_model



Notes:
