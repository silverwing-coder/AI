<h4>Update:  Apr. 2023</h4>

<h2>Environment Set Up for running ML-codes On GPU</h2>

<br/><h3>1. Procedure</h3>

-   Visual Studio (Community Version): Install Tools and Features --> Desktop development with C++ --> install

-   Check the installed GPU hardware version and install appropriate drivers

    -   GPU information: System information --> System summary --> Component --> Display

-   Install CUDA Toolkit

    -   O.S.(Windows), Architecture(x86_64), Version (12.1 in Apr. 2023), Installer Type (exe(local))
    -   Download and install

-   Install cuDNN

    -   O.S.(Windows), Architecture(x86_64), Version (11 in Apr. 2023), Installer Type (exe(local))
    -   Download and unzip
    -   Copy dll files from zipped file folders to CUDA toolkit folders (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v21.1\bin, include, lib)

-   Install Zlib

    -   Download and unzip Zlib package
    -   Add directory path to the directory where <em>zlibwapi.dll</em> is

-   Re-install virtual environment packages
    -   remove torch, torchaudio, torchvision and re-install them by the command from pytorch-cuda dwonload site

<br/><h3>2. Reference</h3>

-   [NVIDIA Deep Learning cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-windows)
