## SNK experiments
Code to reproduce SNK paper and other functional maps stuff.

Configure conda environment.

    conda env create -n snk python=3.8
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
    conda install -c conda-forge meshplot
    pip install -r requirements.txt 

Change path in config.yml