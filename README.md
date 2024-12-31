# Edit3D

:heart: **[2024/12/3]** Due to unforeseen delays, the open-source code will now be released in late December 2024. We are actively working to finalize it, and we appreciate your patience and understanding.

[2024/10/30] Due to unexpected scheduling constraints, the release of the open-source code will be delayed by 2-4 weeks. Thank you for your patience and understanding.

The code will be prepared and open-sourced here by mid-October 2024, in advance of the conference starting on October 28, 2024.


# Gaussian splatting

## Envs 

```bash
conda create --name edit3d python=3.8
conda activate edit3d
pip install --upgrade pip setuptools wheel

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja

pushd GaussianEditor_lib
pip install -r requirements_2.txt
pip install Pillow==9.5.0

pushd gaussiansplatting/submodules
pip install ./diff-gaussian-rasterization
pip install ./simple-knn
pip install easydict
pip install webdataset
pip install albumentations==0.5.2
pip install kornia==0.7.0
pip install diffusers[torch]==0.19.3
pip install rembg
pip3 install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
pip install viser
pip install torch_efficient_distloss
pip install mediapy
pip install plyfile
#pip uninstall torch torchvision
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

popd 
popd
mkdir third_party
pushd third_party
git clone https://github.com/heheyas/viser
pip install --ignore-installed -e viser
popd

pip install mediapipe

pip install git+https://github.com/salesforce/LAVIS.git
pip install transformers==4.25


```
