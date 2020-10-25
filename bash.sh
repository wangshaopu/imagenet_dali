单卡多GPU
python main.py --batch-size 512 --workers 10 -a resnet18 --dist-url 'tcp://127.0.0.1:22451' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /bhome/WangShaoPu/data/ILSVRC2012
python main.py --batch-size 256 --workers 10 -a resnet50 --dist-url 'tcp://127.0.0.1:22451' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /home/WangShaoPu/data/ILSVRC2012

多结点
Node 1
python main.py --workers 10 -a resnet50 --dist-url 'tcp://192.168.123.65:22451' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 /home/WangShaoPu/data

Node 2
python main.py --workers 10 -a resnet50 --dist-url 'tcp://192.168.123.65:22451' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 /home/WangShaoPu/data