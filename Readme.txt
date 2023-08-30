1) Create environment matching environment.yaml, it should contain everything for stylegan3, WarpedGANSpace, GANLatentDiscovery, PDETraversal (also needs kornia) and this project.

2) Clone stylegan3 from https://github.com/NVlabs/stylegan3/tree/main into the folder "stylegan3".
2) a) Copy folders "dnnlib" and "torch_utils" from "stylegan3" to main folder as they are required to run commands from python files in the main folder.

3) Clone WarpedGANSpace from https://github.com/chi0tzp/WarpedGANSpace into the folder "WarpedGANSpace".
3) a) Overwrite files in "WarpedGANSpace" with those in the folder "overwrite".

4) Clone PDETraversal from https://github.com/KingJamesSong/PDETraversal into the folder "PDETraversal".
4) a) Overwrite files in "PDETraversal" with those in the folder "overwrite".


Notes:
This project was tested on one Nvidia Titan X. It may work on multiple GPUs, start by changing the ngpu value in:
overwrite/for WarpedGANSpace/models/gan_load.py and trainer.py
