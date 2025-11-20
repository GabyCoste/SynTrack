Installation instructions for Linux below. Code and install instructions for windows are in the branch called windows_version.

To install:
1. Make sure you have python installed (this code runs for me in python3.7 but may very well work on newer versions as well)
2. Clone this branch of the repo (main branch):
```
git clone https://github.com/GabyCoste/SynTrack.git
```
3. Move to the newly created SynTrack folder:
```
cd SynTrack
```
4. Create a tracked_ims folder and a ims_to_track folder
5. Set up lemon in the SynTrack folder with the following commands:
```
wget https://lemon.cs.elte.hu/pub/sources/lemon-1.3.1.tar.gz
tar -xvzf lemon-1.3.1.tar.gz
cd lemon-1.3.1
mkdir build
cd build/
cmake -DCMAKE_INSTALL_PREFIX="Your_Path_to_Folder/SynTrack/lemon-1.3.1"
make
make install
```
6. In SynTrack_batch_GC.py, edit the path to lemon in lines 419 and 420, here they would respectively be
```
"-I", "Your_Path_to_Folder/SynTrack/lemon-1.3.1/include",
"-L", "Your_Path_to_Folder/SynTrack/lemon-1.3.1/lib",
```
7. Start your python env and run the file: SynTrack_batch_GC.py
```
python 
SynTrack_batch_GC.py
```
Install necessary packages as needed. Add images to track in ims_to_track folder. They should be 4D tif files with the following name convention:

CageID_MouseID_ROI_anything_else.tif

e.g. F14_5_roi4_segmentation_CROP_processed.tif

The output will be a 4D tif file in the tracked_ims folder with the following name: CageID_MouseID_ROI_syntracked.tif (e.g. F14_5_roi4_syntracked.tif)
