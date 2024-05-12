## Exploring Rate of Change of Frequency RoCoF events in the Nordic Syncrenous Area

Welcome to this GitHub repository! 
This project is an integral part of my Master's Thesis at the Norwegian University of Life Sciences (NMBU) and focuses on analyzing and processing frequency data from FinGrid to identify Rate of Change of Frequency (RoCoF) events in the Nordic Synchronous Area. The code have been used to analyse data from FinGrid in the time from 2015-2023. Which has an unzipped size of about 83 GB.

### The repository is organized into two main folders:

Analysis: This folder contains scripts that analyze and process frequency data to detect RoCoF events. The code is somewhat commented on to enhance readability.

Plotting: The scripts in this folder generate visualizations that help illustrate the findings.

### Tools and Techniques

The analysis employs the Whittaker-Henderson smoother. The math behind it was first introduced by Edmund Taylor Whittaker in 1922. The modern implementation of this method is written in Java by Michael Schmid, David Rath, and Ulrike Diebold in the paper "Why and How Savitzky–Golay Filters Should Be Replaced". The smoothing method used in my work is adapted to Python. To explore the original code, see their paper at https://doi.org/10.1021/acsmeasuresciau.1c00054 (the Java code can be found under "Supporting Information")

### Use of the code in this repository

The code in this repository is by no means faultless. It is published under the MIT License, so feel free to explore it and use it in your own research or application.
