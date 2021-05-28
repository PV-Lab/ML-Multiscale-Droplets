# Automated-Fluid-Optimization

**Project Leader:** Aleks Siemenn \<<asiemenn@mit.edu>\>

**Collaborators:** Evyatar Shaulsky, Matthew Beveridge, Tonio Buonassisi, Sara M. Hashmi, Iddo Drori

**Abstract:** Autonomous optimization is a process by which hardware conditions are discovered that generate an optimized experimental product without the guidance of a domain expert. We design an autonomous optimization framework to discover the experimental conditions within fluid systems that generate discrete and uniform droplet patterns. Generating discrete and uniform droplets requires high-precision control over the experimental conditions of a fluid system. Fluid stream instabilities, such as Rayleigh-Plateau instability and capillary instability, drive the separation of a flow into individual droplets. However, because this phenomenon leverages an instability, by nature the hardware must be precisely tuned to achieve uniform, repeatable droplets. Typically this requires a domain expert in the loop and constant re-tuning depending on the hardware configuration and liquid precursor selection. Herein, we propose a computer vision-driven Bayesian optimization framework to discover the precise hardware conditions that generate uniform, reproducible droplets with the desired features, leveraging flow instability without a domain expert in the loop. This framework is validated on two fluid systems, at the micrometer and millimeter length scales, using microfluidic and inkjet systems, respectively, indicating the application breadth of this approach.

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
