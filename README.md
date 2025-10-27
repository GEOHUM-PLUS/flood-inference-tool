To use the tool, first we need to create a virtual environemt using [Anaconda or Miniconda](https://www.anaconda.com/download/success). I recommend using Miniconda, but if you have anaconda already installed that is more than enough. We use the following command on Anaconda prompt to create an environment called ```p3.12``` and install python and GDAL on it.

```
conda create -n flood_tool -c conda-forge python=3.12 gdal pystac-client
```

```
conda activate flood_tool
```

```
pip install -r requirements.txt
```
