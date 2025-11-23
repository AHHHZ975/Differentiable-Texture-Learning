The code for the paper 
# End-to-End Fine-Tuning of 3D Texture Generation using Differentiable Rewards
While recent 3D generative models can produce high-quality texture images, they often fail to capture human preferences or meet task-specific requirements. Moreover, a core challenge in the 3D texture generation domain is that most existing approaches rely on repeated calls to 2D text-to-image generative models, which lack an inherent understanding of the 3D structure of the input 3D mesh object. To alleviate these issues, we propose an end-to-end differentiable, reinforcement-learning-free framework that embeds human feedback, expressed as differentiable reward functions, directly into the 3D texture synthesis pipeline. By back-propagating preference signals through both geometric and appearance modules of the proposed framework, our method generates textures that respect the 3D geometry structure and align with desired criteria. To demonstrate its versatility, we introduce three novel geometry-aware reward functions, which offer a more controllable and interpretable pathway for creating high-quality 3D content from natural language. By conducting qualitative, quantitative, and user-preference evaluations against state-of-the-art methods, we demonstrate that our proposed strategy consistently outperforms existing approaches. We will make our implementation code publicly available upon acceptance of the paper.

# Installation
1- We recommend using a virtual environment or a conda environment.
```
conda create -n texture_dpo python=3.10
```

2- Install the proper version of Pytorch library depending on your machine. For more information see the [Pytorch webpage](https://pytorch.org).
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3- Install other dependencies as follows:
```
pip install tqdm rich omegaconf ninja
pip install numpy scipy scikit-learn kornia matplotlib opencv-python imageio imageio-ffmpeg einops
pip install huggingface_hub diffusers==0.23.1 accelerate transformers                                 ### for stable-diffusion
pip install peft==0.6.2                                                                               ### for Prompt-Tuning
pip install xatlas plyfile pygltflib trimesh rtree                                                    ### for dmtet and mesh export
pip3 install pymeshlab                                                                                ### for remeshing and re-parameterization
pip install pyvista                                                                                   ### for visualizing a mesh ant its symmetry axis
pip install scikit-image                                                                              ### for SSIM metric
pip install dearpygui kiui                                                                            ### for gui
pip install git+https://github.com/NVlabs/nvdiffrast/                                                 ### Nvidia differentiable rasterizer
pip install libigl                                                                                    ### For curvature reward computation
pip install ml-collections absl-py wandb inflect pydantic
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121                           ### for xFormers
```

# Some tips for any potential issuer during or after the installation process
1- For users on Windows 10 or 11, if you have issues with the nvdiffrast package when running the code, it is most probably because of two things:

a) You should install Microsoft Visual Studio 2022 to provide a C++ compiler for the nvdiffrast. This is because the nvdiffrast runs the compiler existing in a directory of Microsoft Visual Studio on Windows. (Reference: https://github.com/NVlabs/nvdiffrast/issues/154)

b) If you have issues related to importing some modules or DLLs (e.g. "no module found" or something like that), this is because Python cannot find some DLLs in Windows and you should add the following line to the beginning of the file you're getting those errors: (Reference: https://github.com/NVlabs/nvdiffrast/issues/46#issuecomment-1910590820)
```
import os
os.add_dll_directory(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\lib\x64")
os.add_dll_directory(r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64")
os.add_dll_directory(r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64")

# c10.dll
os.add_dll_directory(r"C:\Users\User\.conda\envs\texturegen_intex\Lib\site-packages\torch\lib")

# cudart64_12.dll
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")
```
2- If you have any issue with loading the Python modules in the code of D3PO, it is most probably because of the fact that the Python interpreter cannot read Python scripts from other folders in the directory of D3PO project. So, you should manually add the path to the "train_d3po.py" file or any other file that you are running:
```
ROOT_DIR = r'C:\Users\User\Desktop\Texture_DPO\D3PO'
sys.path.insert(0, ROOT_DIR)
```

And remove the following two lines from the script:

```
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
```

3- If you get an error something like ```ImportError: cannot import name 'randn_tensor' from 'diffusers.utils'```, be aware that "randn_tensor" has been moved to another module in the newer version of the diffusers library. So, you need to import it using 
```from diffusers.utils.torch_utils import randn_tensor```
instead of previously used ```from diffusers.utils import randn_tensor```.

4- If you have any issues with the LORA or its versioning, comment out the lines 17-21 in the file "C:\Users\User\.conda\envs\texture_dpo\lib\site-packages\diffusers\utils\deprecation_utils.py"
(Reference: https://github.com/d8ahazard/sd_dreambooth_extension/issues/1456#issuecomment-1932484747)

5- If you have any issues like ```RuntimeError: element 0 of variables does not require grad and does not have a grad_fn```, this is most probably related to the versioning of the diffusers library. The problem is that the latest version of the diffusers is not compatible with the D3PO and InteX code, so you need to re-install the diffusers with the version ```0.23.1```. Also, another possibility could be the ```transformers``` library according to [here](https://github.com/huggingface/transformers/issues/25006). We have installed and tested the ```transformers==4.29.2``` and it works without any problem but it seems that the newer versions of the ```transformers``` library causes this error.

6- If you keep getting this error ```ImportError: cannot import name 'cached_download' from 'huggingface_hub' (/home/ahz/.conda/envs/texture_dpo/lib/python3.10/site-packages/huggingface_hub/__init__.py)```, this is a recent issue in the ```huggingface-hub``` library with the version ```0.26.1```. So there are two solutions for this: First, replace ```cached_download``` by ```hf_hub_download``` in your "C:\Users\User\.conda\envs\texture_dpo\Lib\site-packages\diffusers\utils\dynamic_modules_utils.py" file (this is actually the file that it raises the error) (reference: https://github.com/easydiffusion/easydiffusion/issues/1851#issuecomment-2440911097). Second, install a lower version could be a solution: ```pip install huggingface_hub==0.25.2``` (Reference: https://github.com/huggingface/huggingface_hub/issues/2617)

7- If you get this error ```OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.```, it occurs when a Python program or library (like PyTorch) tries to access CUDA for GPU computation but cannot find the CUDA installation because the ```CUDA_HOME``` environment variable is not set. You need to manually set the ```CUDA_HOME``` variable to point to the root directory of your CUDA installation. Here is how to resolve it:

a) Typically, CUDA is installed in ```/usr/local/cuda``` on Linux or ```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X``` on Windows (where vX.X is your version).

For example: On Linux: ``` /usr/local/cuda ``` and on Windows:  ``` C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8 ```

b) Set Environment Variable: 

Linux/macOS: Open your terminal and add the following lines to your ``` ~/.bashrc ```:
```
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
```
Then run:
```
source ~/.bashrc
```

Windows: First install the CUDA toolkit from [here](https://developer.nvidia.com/cuda-downloads?target_os=Windows). Then, open the Start menu, search for "Environment Variables."
In "System Properties," click "Environment Variables." Under "System variables," click "New," and add:
    Variable Name: ```CUDA_HOME```
    Variable Value: ```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8``` (adjust based on your version).

c) Restart your Terminal or IDE: After setting the environment variable, restart the terminal or the IDE you're using.


# Citation
We appreciate your interest in our research. If you want to use our work, please consider the proper citation format written below.
```

@article{zamani2025geometry,
  title={End-to-End Fine-Tuning of 3D Texture Generation using Differentiable Rewards},
  author={Zamani, AmirHossein and Xie, Tianhao and Aghdam, Amir G and Popa, Tiberiu and Belilovsky, Eugene},
  journal={arXiv preprint arXiv:2506.18331},
  year={2025}
}
```

