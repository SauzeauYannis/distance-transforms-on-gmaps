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