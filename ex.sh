source ~/openvino/openvino_dist/bin/setupvars.sh
python3 human_pose_estimation.py -m intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml -d MYRIAD -at openpose

