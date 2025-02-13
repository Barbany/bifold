# BiFold: Bimanual Cloth Folding with Language Guidance

[`Website`](https://barbany.github.io/bifold/) | [`BibTeX`](#-citation)  | [`arXiv`](https://arxiv.org/abs/2501.16458) | [`Dataset`](https://zenodo.org/records/14851100)

Official implementation of the paper.

**[Oriol Barbany](https://barbany.github.io/), [Adrià Colomé](https://www.iri.upc.edu/staff/acolome) and [Carme Torras](https://www.iri.upc.edu/people/torras/)**

**Institut de Robòtica i Informàtica Industrial (CSIC-UPC), Barcelona, Spain**

**IEEE International Conference on Robotics and Automation (ICRA), 2025**

- [BiFold: Bimanual Cloth Folding with Language Guidance](#bifold-bimanual-cloth-folding-with-language-guidance)
  - [🛠 Installation](#-installation)
    - [🐍 Python](#-python)
    - [🤖 SoftGym simulator](#-softgym-simulator)
  - [👚 Data](#-data)
    - [✋ Unimanual simulated dataset](#-unimanual-simulated-dataset)
    - [👐 Bimanual simulated dataset](#-bimanual-simulated-dataset)
    - [📸 Real-world dataset](#-real-world-dataset)
  - [🏃 Running the code](#-running-the-code)
  - [🏷 License](#-license)
  - [🤝 Acknowledgements](#-acknowledgements)
  - [📚 Citation](#-citation)


## 🛠 Installation

### 🐍 Python

1. Clone this repository:
```
git clone git@github.com:Barbany/bifold.git
```
2. Create a new conda environment and install the `bifold` package and its dependencies:
```
conda create -n bifold python=3.9 -y
conda activate bifold
pip install -e ./bifold
```
3. Make sure PyTorch is correctly installed and CUDA is available:
```
python -c "import torch; print(torch.cuda.is_available())"
```
If CUDA is not available, consider re-installing PyTorch following the official [installation instructions](https://pytorch.org/get-started/locally/).

[Optional] You can also install optional dependencies. For example, the development packages are installed by running:
```
pip install -e ./bifold[dev]
```

> [!NOTE]
> If you are using `zsh`, you may have to use quotes, i.e., `pip install -e ".[dev]"`, or use the `noglob` function, i.e., `noglob pip install -e ./bifold[dev]`

### 🤖 SoftGym simulator

You can skip this installation if you don't want to evaluate the predictions of the model on a simulator.

Follow the steps below even if you have a local SoftGym installation, as we provide custom environments to load CLOTH3D assets for the `Trousers` and `Tshirt` environments.

> [!WARNING]
> Make sure you place the CLOTH3D assets in `datasets/CLOTH3D` or modify the `desps/prepare.sh` path, otherwise PyFlex won't be able to locate them.

1. Install the `sim` optional dependencies of the `bifold` library from the root of the repository:
```
pip install -e ".[sim]"
```

2. Compile the custom PyFlex package included in `deps/PyFlex`. To do that it is recommended to use Docker (see [this blogpost](https://danieltakeshi.github.io/2021/02/20/softgym/) if you want to familiarize yourself with a regular SoftGym installation). Assuming your conda is in `$HOME/miniconda3`, run an interactive session using the Docker image `xingyu/softgym:latest`:
```
docker run -v $PWD:/workspace/bifold -v $HOME/miniconda3:$HOME/miniconda3 -e LOCAL_HOME=$HOME -it xingyu/softgym:latest bash
```

3. **Inside the Docker session**, move to the root, export variables to use PyFlex and import CLOTH3D assets, and compile.
```
cd bifold/
export PATH="$LOCAL_HOME/miniconda3/bin:$PATH"
. ./deps/prepare.sh
. ./deps/compile.sh
```

4. Locate at the root of the repository and set variables permanently in the conda environments (make sure to reactivate environment for the changes to take effect). Alternatively, you can set the variables in your shell, but remember to set them for every new session:
```
conda env config vars set PYFLEXROOT=${PWD}/deps/PyFlex
conda deactivate
conda activate bifold
conda env config vars set PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
conda env config vars set LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
conda env config vars set CLOTH3D_PATH=${PWD}/datasets/cloth3d
conda deactivate
conda activate bifold
```
5. Check that `pyflex` was correctly installed by running
```
python -c "import pyflex"
```

## 👚 Data

For convenience, we uploaded the parsed actions with aligned language instructions as well as the renders with cloth textures in [Zenodo](https://zenodo.org/records/14851100). We include the unimanual, bimanual and real datasets. For the bimanual dataset, make sure to download the [vr-folding dataset](https://huggingface.co/datasets/robotflow/vr-folding) following the instructions below. For the others, you may skip the following details unless you want to create your own dataset.

### ✋ Unimanual simulated dataset

> [!IMPORTANT] 
> There is no need to do anything if you downloaded the dataset from the previous [Zenodo link](https://zenodo.org/records/14851100). 

However, in case you want to re-generate the data, follow the instructions [to generate configurations and expert demonstrations](https://github.com/dengyh16code/language_deformable). Once this is done, the demonstrations will be saved in `raw_data`. Then, you can generate the dataset, which should generate two `.pkl` files for 100 and 1000 demonstrations. Finally, store these files in `datasets/single_data/.`. The configurations can be saved in the SoftGym cache specified in the configuration, which by default sits in `datasets/softgym_cache`.

Since our model with context needs information about the previous actions, we created a script to generate the sequential unimanual dataset. From the root of the repository, run:
```
python scripts/create_unimanual_sequential_dataset.py --use_rgb --task -All --n_demos 100 --save_path_root datasets/single_data_sequential --root PATH/TO/raw_data
```
indicating the path to the previously generated `raw_data` file.

### 👐 Bimanual simulated dataset

Our bimanual dataset relies on the [vr-folding dataset](https://huggingface.co/datasets/robotflow/vr-folding), which you can download from HuggingFace following the instructions. Concatenate all zip files from `folding/` and unzip the resulting file. Then, move `vr_folding_dataset.zarr` to `datasets/folding/.`. There, you should have the other files downloaded from our [link in Zenodo](https://zenodo.org/records/14851100).

If you want to regenerate the actions, run
```
python -m bifold.data.create_dataset_partitions --actions_path /PATH/TO/ACTION_FILES
```

Then, move to `scripts/rendering/` and create the textured CLOTH3D assets by running:
```
python create_textured_objs.py --zarr_root_path /PATH/TO/ZARR_FILE --obj_root_path /PATH/TO/TEXTURED_MESHES --cloth3d_root_path /PATH/TO/CLOTH3D_ASSETS
```

Finally, generate the renders with the following command:
```
python run_all_renders_actions.py --action_root_path /PATH/TO/ACTION_FILES --renders_root_path /PATH/TO/RENDERS --cloth3d_root_path /PATH/TO/CLOTH3D_ASSETS
```

### 📸 Real-world dataset

You can create a new dataset simply with RGB images taken from any camera. To process the dataset, follow the next steps. Our code expects the following file structure (with names in uppercase being placeholders):

```
/PATH/TO/DATA
├── CATEGORY_1
│   └── rgb
│       ├── FILE_NAME_1.png
│       ├── ...
│       └── FILE_NAME_N.png
└── CATEGORY_2
    └── rgb
        ├── FILE_NAME_1.png
        ├── ...
        └── FILE_NAME_M.png
...
```
Each category can also have additional subfolders (or modalities), e.g., `depth/`, `raw_rgb/`, and `raw_depth/`. These categories are recognized by the cropping script and in case they exist, cropped versions of the files in them will be created.

1. Create segmentation masks. We use the [segment anything](https://github.com/facebookresearch/segment-anything) model and provide point prompts. You can run
```
python scripts/create_masks.py --checkpoint /PATH/TO/CHECKPOINT --path_to_data /PATH/TO/DATA
```
This will generate the binary masks as well as overlay images to see the points and the mask on top of the image. In case the segmentation is not as expected, you can modify the input points in `scripts/create_masks.py`.

2. Create crops:
```
python create_crops_w_mask.py --path_to_data /PATH/TO/DATA
```
This will create cropped versions of all the modalities in the data path.

3. [Optional] If you want to annotate the images, you can use our annotation pipeline. To do so, install our fork of `ipyannotations` in which we define the `PointAnnotator`, by running `pip install deps/ipyannotations`. Then, launch a Jupyter Notebook session and open `scripts/bimanual/annotate_actions.ipynb`.

## 🏃 Running the code

> [!IMPORTANT]  
> Make sure you correctly indicated your dataset root in the `dataset_root` parameter of the configuration in `bifold/conf/config.yaml`, which by default is at `$HOME/bifold/datasets`. Also verify that the SoftGym cache (`softgym_cache`) and the output directory (`hydra.run.dir`) are fine.

To perform training followed by evaluation using the default BiFold parameters, simply run:
```
python -m bifold
```

If you want to evaluate a model, run:
```
python -m bifold eval_only=true
```
using the arguments to describe the model you want to load.

BiFold uses [Hydra configuration](https://hydra.cc/). Simply put, this takes the base configuration specified in the decorator of your main function (`bifold/conf/config.yaml`) and composes different configurations, e.g., it takes one model from `bifold/conf/model/`, one optimizer from `bifold/conf/optim/`, etc. This allows to isolate configurations but also to easily switch modules, e.g., by running:
```
python -m bifold optim=adamw
```
you can use the AdamW optimizer instead of the default one.


Make sure to familiarize yourself with the Hydra framework if you have any doubt. The composed configuration will be saved in a file named `config.yaml` of your output directory.


## 🏷 License

The code in this repository is released under the MIT license as found in the [LICENSE](LICENSE) file.
The custom `PyFlex` code is included with its [original license](./deps/PyFlex/LICENSE.txt).

## 🤝 Acknowledgements

- Some parts of this code are based on https://github.com/dengyh16code/language_deformable.
- For the CLIP experiments, we adapt the code from https://github.com/openai/CLIP.
- The annotation pipeline for real images is based on https://github.com/janfreyberg/ipyannotations.

## 📚 Citation

If you find this work useful, please cite our work:
```
@misc{barbany2025bifold,
    title={{BiFold: Bimanual Cloth Folding with Language Guidance}}, 
    author={Oriol Barbany and Adrià Colomé and Carme Torras},
    year={2025},
    eprint={2501.16458},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2501.16458}, 
}
```