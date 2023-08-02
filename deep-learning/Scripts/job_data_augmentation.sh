 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J myJob
 #BSUB -n 1
 #BSUB -W 24:00
 #BSUB -R "rusage[mem=32GB]"
 #BSUB -R "select[gpu32gb]"
 #BSUB -o logs/%J.out
#  BSUB -e logs/%J.err
 module load python3/3.9.11
 module load cuda/8.0
 module load cudnn/v7.0-prod-cuda8
 module load ffmpeg/5.0.1

 
#  pip3 install --user datasets>=1.18.3
#  pip3 install --user transformers==4.11.3
#  pip3 install --user librosa
#  pip3 install --user jiwer
#  pip3 install --user evaluate
#  pip3 install --user rouge_score

 echo "Running script..."
 python3 /zhome/bd/4/181258/code/deep_learning/data_augmentation.py
