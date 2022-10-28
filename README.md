# mask-detection

## How to run
### To train
```
python mask-detection-nn.py  --init-LR <init-LR> --epochs <epochs> --batch-size <batch-size> --dataset-path <path to your datasets>
```

### To demo
```
python camera.py  --proto-path <path to your prototxt> --weights-path <path to your weights> --model-path <path to your model>
```