The objective of the project is to combine different shots of the same scene with different opening times, in order to create a high quality image, which should not have been under-exposed or used in the HDR assembly.

The technique used is based on taking images of the same images below different exposure levels and recognizing, for each person, the required image and finding the best exposure using three criteria (contrast, saturation and exposure).

This method is different from the HDR method in that it works directly with displayable images, and does not require a mapping to adjust the dynamics. Not using the HDR method to obtain a high quality image speeds up the taking of the picture, as the process of calibrating the camera to the scene takes place over a long period of time using this method of exposure fusion. This method allows to merge an image without flash (low exposure level) and with flash (high exposure level) unlike HDR. 

The principle is based on laplacian and gaussian pyramids. The method follows the instructions in the document exposure_fusion.pdf written by Tom Mertens, Jan Kautz and Frank Van Reeth.

How to use this program:
1. Put all the images to fuse into a single folder. All images should be the same size and in JPG format
2. Put the folder of images to merge in the 'sample' folder
3. Run the program fuse.py and enter the name of the folder when required (for example, for the 'arch' folder in the 'sample' folder, just enter arch)
4. After a few seconds, the program has finished running and the resulting merged image is directly in the folder containing all the other images. This file ends with the extension '_fused.png' (for example, for the 'arch' folder, it is called 'arch_fused.png' and it is located in the 'sample/arch/' folder)
