# distance-transforms-on-gmaps

## Distance transforms on n-Gmaps

### Wave propagation on bimary n-Gmaps

To compute the distance transform of a binary n-Gmaps, we can use the wave propagation algorithm.

This algorithm is implemented with `wave_propagation_dt_gmap` function in the `wave_propagation` module in the `distance_transform` package.

This function takes one mandatory argument `gmap` that is a binary n-Gmap, and two optional arguments `seeds_identifiers` and `accumulation_directions` that are the seeds and the directions (alphas of the n-Gmap) of the wave propagation. If `seeds_identifiers` is not provided, the algorithm will use the darts with label equal to 0 of the n-Gmaps as the seed. If `accumulation_directions` is not provided, the algorithm will use all the alphas of the n-Gmap as the directions.

5 by 4 n-Gmap distance transform example:

```python
from combinatorial.pixelmap import LabelMap
from combinatorial.zoo_labels import str2labels
from distance_transform.wave_propagation import wave_propagation_dt_gmap

# Create a 5 by 4 n-Gmap that represents an image with 3 lit pixels
G2_IMG = """\
w . . . .
w . . . .
. . . w .
. . . . .
"""

image = str2labels (G2_IMG)

lm_img = LabelMap.from_labels(image)

# Plot the distance transform before the wave propagation
lm_img.plot_dt()
```

![png](docs/images/output_gmap_before_dt.png)
