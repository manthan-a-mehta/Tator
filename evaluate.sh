VERSION=v1_eval
DS=fish_regression #breakfast 50salads gtea
SP=1
GPU_ID=0
version=15
LOGFILE=logs/exp_${VERSION}_${SP}_${DS}.log

CUDA_VISIBLE_DEVICES=${GPU_ID} python3.7 eval.py "/home/fish/manthan/Tator/swin_tiny_patch4_window7_224/default/version_${version}/checkpoints/best_loss.ckpt"   > "$LOGFILE" 2>&1 &