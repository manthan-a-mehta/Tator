echo "using image size of 100 to train"
VERSION=v3_100
DS=fish_regression #breakfast 50salads gtea
SP=1
GPU_ID=2
LOGFILE=logs/exp_${VERSION}_${SP}_${DS}.log

CUDA_VISIBLE_DEVICES=${GPU_ID} python3.7 main.py  > "$LOGFILE" 2>&1 &