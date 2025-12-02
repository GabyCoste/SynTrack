
Installation instructions for Windows below. Code and install instructions for windows are in this branch called windows_version.
In order to install lemon I had to use a VSCode Powershell terminal because some of the commands were not available in the normal windows command prompt. It's possible it would work in a windows powershell terminal too but I haven't tried that yet. 

To install:
1. Make sure you have python installed (this code runs for me in python3.7 but may very well work on newer versions as well)
2. Clone this branch of the repo (main branch):
```
git clone --single-branch --branch windows_version https://github.com/GabyCoste/SynTrack.git
```
3. Move to the newly created SynTrack folder:
```
cd .\SynTrack\
```
4. Create a tracked_ims folder and a ims_to_track folder
5. Set up lemon in the SynTrack folder with the following commands:
```
iwr https://lemon.cs.elte.hu/pub/sources/lemon-1.3.1.tar.gz -OutFile lemon-1.3.1.tar.gz
tar -xvzf lemon-1.3.1.tar.gz
cd .\lemon-1.3.1\
mkdir build
cd .\build\
cmake ..
cmake --build . --config Release
cmake --install . --config Release
cmake .. -DCMAKE_INSTALL_PREFIX="Your_Path_to_Folder\SynTrack\lemon-1.3.1"
cmake --install . --config Release
```
Yes I realize some of these steps might be redundant but that is what I ran and it worked so just want to make sure.

6. In SynTrack_batch_GC.py, edit the path to lemon in lines 430 and 432, here they would respectively be
```
"/I", r"Your_Path_to_Folder\SynTrack\lemon-1.3.1\include",

"/LIBPATH:Your_Path_to_Folder\\SynTrack\\lemon-1.3.1\\lib",
```
7. Activate your python venv
8. Point the current terminal to our C++ compiler, in my case it is:
```
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
```
You will have to run the above command every time you start a new terminal.

9. Start tracking by running the file: SynTrack_batch_GC.py
```
python SynTrack_batch_GC.py
```
Install necessary packages as needed. Add images to track in ims_to_track folder. They should be 4D tif files with the following name convention:

CageID_MouseID_ROI_anything_else.tif

e.g. F14_5_roi4_segmentation_CROP_processed.tif

The output will be a 4D tif file in the tracked_ims folder with the following name: CageID_MouseID_ROI_syntracked.tif (e.g. F14_5_roi4_syntracked.tif)
