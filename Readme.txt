1) Create environment matching environment.yaml, it should contain everything for stylegan3, WarpedGANSpace and GANLatentDiscovery

2) Clone stylegan3 from https://github.com/NVlabs/stylegan3/tree/main into the folder "stylegan3"
2) a) Copy folders dnnlib and torch_utils from stylegan3 to main folder

3) Clone WarpedGANSpace from https://github.com/chi0tzp/WarpedGANSpace into the folder "WarpedGANSpace"
3) a) In lib/__init__ and lib/trainer rename imports from .aux to ._aux to link to the correct file
3) b) Run python download_models.py to download pretrained models
3) c) Overwrite files in WarpedGANSpace with those in the folder overwrite



Notes:
It is written for one graphics card and tested on Nvidia Titan X.
