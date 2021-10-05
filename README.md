# Automated-Fluid-Optimization

**Project Leader:** Aleks Siemenn \<<asiemenn@mit.edu>\>

**Collaborators:** Evyatar Shaulsky, Matthew Beveridge, Tonio Buonassisi, Sara M. Hashmi, Iddo Drori

**Abstract:** Generating droplets from a continuous stream of fluid requires precise tuning of a device to find optimized control parameter conditions. Furthermore, as the length scale of the fluid flow changes, the formation physics and optimized conditions that induce flow decomposition into droplets also change. Hence, current physics-based droplet optimization tools are sensitive to the scale of the droplet being optimized and also require _a priori_ knowledge of the system physics to perform optimization. We propose a Bayesian machine learning tool integrated with computer vision to optimize droplets within various devices at multiple length scales while requiring no prior domain knowledge of the system. This control method is validated on two devices of different length scales and different droplet formation physics: an inkjet device at the milliscale and a mircofluidics device at the microscale.

**Github Repo:** \<<https://github.com/PV-Lab/Automated-Fluid-Optimization>\>

**Location of data:**

[1] Externally available: DOI 10.17605/OSF.IO/U4KHQ \<<https://osf.io/u4khq/>\>

[2] Internally available: Dropbox (MIT)\Buonassisi-Group\ASD Team\Archerfish\05_Data\Imaged_droplets

**Sponsors:** C2C, Google

*******

## Explanation of code within GitHub Repo:

### [1] Automated_BO.ipynb
Runs an automated Bayesian optimization procedure. Takes input conditions, computes an objective value score via image processing and computer vision, and then outputs new, optima conditions based on the selected decision policy.

### [2] Plot_Figures.ipynb
Runs all figure plotting for the inkjet and microfluidics systems used in this study using data from OSF.IO.
