# ML-Multiscale-Droplets

**Project Leader:** Alexander (Aleks) E. Siemenn \<<asiemenn@mit.edu>\>

## When using the code for any scientific publications or conferences, please cite our research article as:

## **Alexander E. Siemenn, Evyatar Shaulsky, Matthew Beveridge, Tonio Buonassisi, Sara M. Hashmi, and Iddo Drori, "A Machine Learning and Computer Vision Approach to Rapidly Optimize Multiscale Droplet Generation", ACS Applied Materials & Interfaces 2022 14 (3), 4668-4679. DOI: 10.1021/acsami.1c19276**

**Collaborators:** Evyatar Shaulsky, Matthew Beveridge, Tonio Buonassisi, Sara M. Hashmi, Iddo Drori

**Abstract:** Generating droplets from a continuous stream of fluid requires precise tuning of a device to find optimized control parameter conditions. Furthermore, as the length scale of the fluid flow changes, the formation physics and optimized conditions that induce flow decomposition into droplets also change. Hence, current physics-based droplet optimization tools are sensitive to the scale of the droplet being optimized and also require _a priori_ knowledge of the system physics to perform optimization. We propose a Bayesian machine learning tool integrated with computer vision to optimize droplets within various devices at multiple length scales while requiring no prior domain knowledge of the system. This control method is validated on two devices of different length scales and different droplet formation physics: an inkjet device at the milliscale and a mircofluidics device at the microscale.

**Github Repo:** \<<https://github.com/PV-Lab/ML-Multiscale-Droplets>\>

**Location of data:**

[1] Externally available: DOI 10.17605/OSF.IO/U4KHQ \<<https://osf.io/u4khq/>\>

[2] Internally available: Dropbox (MIT)\Buonassisi-Group\ASD Team\Archerfish\05_Data\Imaged_droplets

**Sponsors:** C2C, Google

*******

## Explanation of code within GitHub Repo:

### [1] utilization.ipynb
Python notebook that utilizes all image analysis, computer vision, and Bayesian optimization code to compute the new predicted optimal control parameter values. Run this code using the directions notebook on your own data or the externally available data.

### [2] crop.py
Reads in the droplet images, crops them, and rotates them.

### [3] segmentation.py
Runs watershed segmentation on the droplets images to segmented and index them apart from the background pixels.

### [4] loss.py
Computes the yield and geometric losses of the droplet image. Specify your own loss functions here, if you wish.

### [5] bo.py
Runs Bayesian optimization on the processed and labeled droplet data. The input variable "data" should contain the set x &#8712; X<sup>(N)</sup>, where x are the normalized values of N device control parameters, as well as the computer vision-compute loss scores. The N device control parameters are arbitrary, such that they can be specified by the user based on the user's specific device hardware. New predicted condtions will be output and can be saved to your local computer as a csv.
