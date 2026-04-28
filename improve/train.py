yolo task=detect mode=train model=/home/dl/xgt/improve/ultralytics/cfg/models/11/yolo11s.yaml data=/home/dl/xgt/improve/myCoco.yaml epochs=300 device=0,1,2,3,4,5,6,7 batch=64 patience=300 save_json=True workers=16


export PYTHONPATH="/home/dl/xgt/improve/ultralytics:$PYTHONPATH"


cd /home/dl/xgt/improve/ultralytics
pip uninstall -y ultralytics
pip install -e .

