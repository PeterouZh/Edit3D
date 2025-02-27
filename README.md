# Edit3D

# Edit 3D gaussian splatting

## Envs 

```bash
conda create --name edit3d python=3.8
conda activate edit3d
pip install --upgrade pip setuptools wheel

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja

pushd edit_3dgs/GaussianEditor
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

pip install gradio==3.50.2
pip install -U git+https://github.com/IDEA-Research/GroundingDINO.git
pip install segment-anything==1.0
pip install onnxruntime==1.16.3
#pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
pip install viser
pip install torch_efficient_distloss
pip install mediapy
pip install plyfile

popd 
popd
mkdir third_party
pushd third_party
git clone https://github.com/heheyas/viser
pip install --ignore-installed -e viser
popd

pip install mediapipe
pip install spacy==3.7.5
pip install git+https://github.com/salesforce/LAVIS.git@v1.0.1
pip install transformers==4.25
pip install huggingface-hub==0.23.3

pip install pySciTools

```

## Training

## Download datasets 

[CACHE_DIR.zip](https://1drv.ms/u/c/5240e76ec7fdaf3d/Ee46eLUa2z5MousMFtr5Zq0BaVcpIvREKeDs5O8fqmGoEQ?e=zljr3U)

```bash
unzip CACHE_DIR.zip
```


### Edit 3dgs face

```bash
export CUDA_VISIBLE_DEVICES=0
python -c "from scripts.test_edit3d import Testing_edit3dgs;\
      Testing_edit3dgs().test_edit3dgs_face(debug=False)"
      
```

<img src=".data/face.gif" width=600>

### Edit 3dgs face & hair

```bash
export CUDA_VISIBLE_DEVICES=0
python -c "from scripts.test_edit3d import Testing_edit3dgs;\
      Testing_edit3dgs().test_edit3dgs_face_hair(debug=False)"

# convert -delay 30 -loop 0 save/it1500-test/*.png face_hair.gif
```

<img src=".data/face_hair.gif" width=600>