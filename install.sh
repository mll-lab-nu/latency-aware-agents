conda create -n inference_engine python=3.11 -y
conda activate inference_engine

python -m pip install --upgrade pip setuptools wheel
conda install -c conda-forge cmake -y

pip install -r requirements.txt
AutoROM --accept-license