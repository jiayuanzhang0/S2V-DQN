# Introduction
This is a PyTorch implementation of S2V-DQN for solving graph combinational optimization problem.  
Paper: Khalil, E., Dai, H., Zhang, Y., Dilkina, B., & Song, L. (2017). Learning combinatorial optimization algorithms over graphs. Advances in neural information processing systems, 30.

# Installation  
conda create --name S2V-DQN python=3.8 -y
conda activate S2V-DQN

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib

cd S2V-DQN
pip install -r requirements.txt

# Train
python train.py

# Test
python test.py



