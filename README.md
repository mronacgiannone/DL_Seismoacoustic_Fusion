# DL_Seismoacoustic_Fusion
The information in this repository outlines how to extract and process the seismic and infrasound data used to train both seismic and seismoacoustic neural networks as introduced in the manuscript, "Deep Multimodal Learning for Seismoacoustic Fusion to Improve Earthquake-Explosion Discrimination within the Korean Peninsula", (Ronac Giannone et al., 2024, in review). ObsPy and TensorFlow software packages are used for geophysical and deep learning analyses, respectively.  
## Install
To install:
conda env create -f environment.yml
## Activate
To activate:
source activate koreageonet
## Data
The data used in this study can be found at 10.5281/zenodo.10795252.
## Additional Info
Array locations can be found in Korea_Array_Locations.zip. Earthquake-Explosion databases as well as spreadsheets containing information on individual array detections can be found in Spreadsheets.zip.
