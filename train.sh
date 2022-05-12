VERSION=v1
DS=fish_regression #breakfast 50salads gtea
SP=1
ID=1
LOGFILE=logs/exp_${VERSION}_${SP}_${DS}.log

CUDA_VISIBLE_DEVICES=${ID} python3.7 main.py  > "$LOGFILE" 2>&1 &