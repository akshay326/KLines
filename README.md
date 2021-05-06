# KLines

- Find how the OSM China road network data was prepared here: https://colab.research.google.com/drive/136cXt608dnbS-j8LJRFwf58i5-IiXnst?usp=sharing
- See demo of this package here: https://colab.research.google.com/drive/1bZOiGcZHl1P3VAg-UAihTpjDrdn4r90s?usp=sharing
- Datasets in `data` folder are converted from R's `rda` format to NumPy compatible `npz`. They were fetched from FCPS benchmark - https://github.com/Mthrun/FCPS. Article https://www.sciencedirect.com/science/article/pii/S2352340920303954
- Parallel implementation inspired from (KLines means)[https://github.com/YairMarom/k_lines_means/blob/master/k_line_mean_NIPS.pdf] by Marom and Feldman
- np.version.full_version == '1.16.5', later revisions hv slower array lookups

## Data
- Download preprocessed snapshot of the test data here: https://drive.google.com/file/d/1-2qtR2armEKvDlrb6xmazsm9knENJMR6/view?usp=sharing
- You'll need to download and place above snapshot `data` folder for `test_klines_offline.py` to work

## Testing and running benchmarks
- Run `python -m unittest discover` in the root folder

> Part of Master's project, IIT (BHU) Varanasi, 2020-21