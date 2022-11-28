# ML-Multiscale-Droplets

If you use this package for your research, please cite the following paper **Citation:** 

@article{doi:10.1021/acsami.1c19276,
author = {Siemenn, Alexander E. and Shaulsky, Evyatar and Beveridge, Matthew and Buonassisi, Tonio and Hashmi, Sara M. and Drori, Iddo},
title = {A Machine Learning and Computer Vision Approach to Rapidly Optimize Multiscale Droplet Generation},
journal = {ACS Applied Materials \& Interfaces},
volume = {14},
number = {3},
pages = {4668-4679},
year = {2022},
doi = {10.1021/acsami.1c19276},
    note ={PMID: 35026110},
URL = { 
        https://doi.org/10.1021/acsami.1c19276 },
eprint = { 
        https://doi.org/10.1021/acsami.1c19276    }}

### When using the code for any scientific publications or conferences, please cite our research article as:

#### **Alexander E. Siemenn, Evyatar Shaulsky, Matthew Beveridge, Tonio Buonassisi, Sara M. Hashmi, and Iddo Drori, "A Machine Learning and Computer Vision Approach to Rapidly Optimize Multiscale Droplet Generation", ACS Applied Materials & Interfaces 2022 14 (3), 4668-4679. DOI: 10.1021/acsami.1c19276**

**Collaborators:** Evyatar Shaulsky, Matthew Beveridge, Tonio Buonassisi, Sara M. Hashmi, Iddo Drori

**Abstract:** Generating droplets from a continuous stream of fluid requires precise tuning of a device to find optimized control parameter conditions. It is analytically intractable to compute the necessary control parameter values of a droplet-generating device that produces optimized droplets. Furthermore, as the length scale of the fluid flow changes, the formation physics and optimized conditions that induce flow decomposition into droplets also change. Hence, a single proportional integral derivative controller is too inflexible to optimize devices of different length scales or different control parameters, while classification machine learning techniques take days to train and require millions of droplet images. Therefore, the question is posed, can a single method be created that universally optimizes multiple length-scale droplets using only a few data points and is faster than previous approaches? In this paper, a Bayesian optimization and computer vision feedback loop is designed to quickly and reliably discover the control parameter values that generate optimized droplets within different length-scale devices. This method is demonstrated to converge on optimum parameter values using 60 images in only 2.3 hours, $30\times$ faster than previous approaches. Model implementation is demonstrated for two different length-scale devices: a milliscale inkjet device and a microfluidics device.

**Github Repo:** \<<https://github.com/PV-Lab/ML-Multiscale-Droplets>\>

**Location of data:**

Externally available: DOI 10.17605/OSF.IO/U4KHQ \<<https://osf.io/u4khq/>\>

**Sponsors:** C2C, Google

*******

## Explanation of code within GitHub Repo:

### [1] examples.ipynb
Python notebook that utilizes all image analysis, computer vision, and Bayesian optimization code to compute the new predicted optimal control parameter values. Run this code using the directions notebook on your own data or the externally available data.

### [2] crop.py
Reads in the droplet images, crops them, and rotates them.

### [3] segmentation.py
Runs watershed segmentation on the droplets images to segmented and index them apart from the background pixels.

### [4] loss.py
Computes the yield and geometric losses of the droplet image. Specify your own loss functions here, if you wish.

### [5] bo.py
Runs Bayesian optimization on the processed and labeled droplet data. The input variable "data" should contain the set x &#8712; X<sup>(N)</sup>, where x are the normalized values of N device control parameters, as well as the computer vision-compute loss scores. The N device control parameters are arbitrary, such that they can be specified by the user based on the user's specific device hardware. New predicted condtions will be output and can be saved to your local computer as a csv.

