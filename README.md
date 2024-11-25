## InSAR Calculation
This code calculates the azimuth projection (fault parallel) InSAR velocities using two different look directions observation of Sentinel-1 .HDF5 data from the Southern California Earthquake Center (SCEC) Community Geodetic Model (CGM) (Floyd et al., 2023). InSAR look vectors are calculated using Kathryn Materna's [Tectonic_Utils](https://github.com/kmaterna/Tectonic_Utils) with the following equation (Fialko et al., 2001).

$$d_{los} =[U_n  sin⁡(ϕ)-U_e  cos⁡(ϕ) ]⋅sin⁡(λ)+U_u cos⁡(λ)$$

Then, the fault parallel deformation velocity projection is calculated with the following equations (Lindsey et al., 2014):

$$\begin{bmatrix}
LOS_{u1} & LOS_{f1} \\
LOS_{u2} & LOS_{f2}
\end{bmatrix}$$

$$\begin{pmatrix}
V_{u1}\\
V_{u2}
\end{pmatrix} = G \begin{pmatrix}
V_{u1}\\
V_{u2}
\end{pmatrix}$$

$$\begin{pmatrix}
V_{f}\\
V_{u}
\end{pmatrix} = G^{-1} \begin{pmatrix}
V_{1}\\
V_{2}
\end{pmatrix}$$

Additionally, several functions to compare InSAR (or any grid files) are also available including reassigning reference pixels and resampling for different data sources.

## Installation and Usage
### Requirements:
- Python 3.9+
- Kathryn Materna's [Tectonic_Utils](https://github.com/kmaterna/Tectonic_Utils) and [InSAR_CGM_readers_writers](https://github.com/kmaterna/InSAR_CGM_readers_writers)
- [PyGMT](https://www.pygmt.org/latest/)
- Standard Python libraries: numpy, pandas, and Xarray

## References
- Blewitt, G., Hammond, W., & Kreemer, C. (2018). Harnessing the GPS Data Explosion for Interdisciplinary Science. Eos, 99. https://doi.org/10.1029/2018EO104623
- Fialko, Y., Simons, M., & Agnew, D. (2001). The complete (3‐D) surface displacement field in the epicentral area of the 1999 Mw 7.1 Hector Mine Earthquake, California, from space geodetic observations. Geophysical Research Letters, 28(16), 3063–3066. https://doi.org/10.1029/2001GL013174
- Lindsey, E. O., & Fialko, Y. (2016). Geodetic constraints on frictional properties and earthquake hazard in the Imperial Valley, Southern California. Journal of Geophysical Research: Solid Earth, 121(2), 1097–1113. https://doi.org/10.1002/2015JB012516
