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

image = str2labels(G2_IMG)

lm_img = LabelMap.from_labels(image, need_to_save_labels=False)

# Plot the distance transform before the wave propagation
lm_img.plot_dt()

seeds_identifiers = [
    0, 1, 2, 3, 4, 5, 6, 7,                 # First lit pixel
    40, 41, 42, 43, 44, 45, 46, 47,         # Second lit pixel
    104, 105, 106, 107, 108, 109, 110, 111  # Third lit pixel
]

wave_propagation_dt_gmap(lm_img, seeds_identifiers)

# Plot the distance transform after the wave propagation
lm_img.plot_dt()
