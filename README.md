1. conda create -n upright python=3.8    # create new conda environment
2. conda activate upright #activate the env
3. pip install -e . # install local package
4. pip install -r requirements.txt # install requirements.txt
### download ycb datasets
5. python scripts/download_ycb.py
### generate image : It'll be saved in data/images
6. python scripts/generate_data.py
