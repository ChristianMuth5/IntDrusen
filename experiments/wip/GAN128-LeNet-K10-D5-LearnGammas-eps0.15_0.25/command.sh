#!/usr/bin/bash
WarpedGANSpace/train.py --gan-type=GAN128 --reconstructor-type=LeNet --learn-gammas --num-support-sets=10 --num-support-dipoles=5 --min-shift-magnitude=0.15 --max-shift-magnitude=0.25 --batch-size=8 --max-iter=200000
