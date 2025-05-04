<!-- conda create -n experiment python=3.10
conda activate experiment

-i https://pypi.mirrors.ustc.edu.cn/simple

pip install torch==2.5.1 torchvision torchaudio -i https://pypi.mirrors.ustc.edu.cn/simple -->


conda create -n openr1 python=3.11
conda activate openr1
<!-- git clone https://github.com/huggingface/open-r1.git -->
<!-- cd open-r1 -->

pip install transformers trl lighteval -i https://mirrors.aliyun.com/pypi/simple/

pip install /share/leozhilin/experiment/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install qwen-vl-utils[decord]==0.0.8 -i https://mirrors.aliyun.com/pypi/simple/

pip install wandb -i https://mirrors.aliyun.com/pypi/simple/

pip install datasets -i https://mirrors.aliyun.com/pypi/simple/

conda install cuda-nvcc -c conda-forge
cd open-r1
pip install -e ".[dev]" -i https://mirrors.aliyun.com/pypi/simple/

pip install torchvision
