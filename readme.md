### Preprocessing Data

The data is given as csv-files in data/raw/sequences, where each csv is a single sequence.

To merged them all in big tables, please run `python src\concat_sequences.py data\raw\sequences data\raw\tables`

#### Data Analysis

#### Data cleaning

For the ML Model to be useful, the data must be in a good shape. This means that missing values, sensor glitches and sensor inaccuracies must be handled correctly.

Because we have human motion data, we expect to have smooth and continuous data, just like humans move.

Unfortunately, motion tracking systems are not always very accurate and therefore, we need to clean the data.

##### Outlier Detection

###### TODO

1. Visualize different smoothing algorithms:
   a. Implement all smoothing algorithms in a clean, modular way. Algorithms: Kalmar Filter, Savitzky-Golay Filter, Woltring's spline filtering.
   b. Visualize all of them with different parameters and calculate MSE, mean, std.
   c. For every algorithm, choose best paramters (according to visual, MSE, mean and std).
   d. Visualize arm movement in animation with original data, interpolated data, and each of the algorithmns.
   e. Choose best algorithms based on the best, cleanest movement while still being correct.

2. Detect & Remove Outliers:
   a. First, smooth the data with the chosen algorithm (or fit a curve).
   b. Then, calculate the distance from the smoothed velocity curve to the current unsmoothed velocity point.
   c. If this distance is too big, we mark this point as an outlier.
   d. These outliers can be replaced with the smoothed point.
   e. After replacements, smooth the new data.

According to a study by He and Tian, motion trajectory tracking can be improved by using a statistical smoothness measure to eliminate outliers in the data [(He & Tian, 1989)](https://www.sciencedirect.com/science/article/pii/S0167945797000298). This approach is particularly effective in applications involving biomechanical control systems.


###


### References

1. He, J., & Tian, C.-X. (year). _A statistical smoothness measure to eliminate outliers in motion trajectory tracking_. Whitaker Center for Neuro-Mechanical Control, Chemical, Bio and Materials Engineering, Arizona State University, ECG 202, MS-6006, Tempe, AZ 85287-6006, USA.
