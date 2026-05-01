git clone https://github.com/ZihanWang314/latency-sensitive-bench
cd latency-sensitive-bench
git checkout sample-factory-backend

git config -f .gitmodules submodule.flappy-bird-gymnasium.url https://github.com/mindorigin150/flappy-bird-gymnasium.git
git submodule sync --recursive
git submodule update --init --recursive

conda create -n latency python=3.10 -y
conda activate latency
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
wandb login