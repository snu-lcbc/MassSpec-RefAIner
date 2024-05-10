[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Refining EI-MS library search results through atomic-level insights
> Ucak U.V., Ashyrmamatov I., Lee J. Preprint ChemRxiv: [10.26434/chemrxiv-2024-vrqzf](https://doi.org/10.26434/chemrxiv-2024-vrqzf)

Mass spectral reference libraries are fundamental tools for compound identification in electron-ionization mass spectrometry (EI-MS). 
However, the inherent complexity of mass spectra and the lack of direct correlation between spectral and structural similarities present significant challenges in structure elucidation and accurate peak annotation. 
To address these challenges, we have introduced an approach combining CFM-EI, a fragmentation likelihood modeling tool in EI-MS data, with a multi-step complexity reduction strategy for mass-to-fragment mapping. 
Our methodology involves employing modified atomic environments to represent fragments of super small organic molecules and training a transformer model to predict the structural content of compounds based on mass and intensity data. 
This holistic solution not only aids in interpreting EI-MS data by providing insights into atom types but also refines cosine similarity rankings by suggesting inclusion or exclusion of specific atom types.
Tests conducted on EI-MS data from the NIST database demonstrated that our approach complements conventional methods by improving spectra matching through an in-depth atomic-level analysis.

<!-- ![Preprocessing Workflow](./assets/MSpaper_figure2.png) -->
<img src="assets/MSpaper_figure2.png" alt="Preprocessing Workflow" width="600" />
Fragmentation and multi-step complexity reduction plan for EI-MS data interpretation (a) Schematic representation of the data processing workflow, beginning with EI-MS data selection from the NIST Main Library, focusing on compounds with Mw $\leq$ 400 Da, followed by fragment annotation using CFM-EI, and subsequent ion collection. The bottom panel illustrates the initial reduction applied to the pool of fragment ions via similarity thresholding using the Tanimoto coefficient at ECFP2 level. (b) Frequency-based data filtering with respect to atom types (depicted as SMARTS), followed by the process of customizing atomic environment representations to suit analytical needs. The spider chart and adjacent table detail the modifications to AE mappings and the criteria for isotopic abundance-based intensity cutoffs, essential for elements such as S, Cl, and Br.

<!-- ![Model Overview](./assets/MSpaper_figure4.png) -->
<img src="assets/MSpaper_figure4.png" alt="Model Overview" width="600" />
Schematic of the transformer model for converting EI-MS spectral data into structural information. Peak intensities are encoded as logRanks and combined with m/z values as inputs to the transformer encoder. The decoder then predicts structural content of fragment ions and molecular content as reduced atomic environments (rAEs). The histogram displays the model's accuracy, recall, and precision metrics.

<hr style="background: transparent; border: 0.2px dashed;"/>

## Code Usage:
After installing required packages, you can run the code as follows:
```bash
python predict.py --spectrum_file spectrum_sample.txt --preprocess --output results.json
```
<!-- '--spectrum_file', type=Path, required=True, help="The file path where the input spectra can be found. It expects a list of peaks 'mass\tintensity' delimited by lines.") -->
The `--spectrum_file` file should contain a list of peaks with mass and intensity values separated by a tab. The `--preprocess` flag is used to preprocess the raw spectrum data. If it's already preprocessed, `--preprocess` no needed. The results will be saved in the `--output` file. 


To see the full list of options, run:
```bash
python predict.py --help```
