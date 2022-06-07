from cv2 import imread
from combinatorial.pixelmap import LabelMap
from distance_transform.wave_propagation import generalized_wave_propagation_gmap
from data.labels import labels
from distance_transform.dt_utils import build_dt_grey_image_from_gmap, plot_dt_image

image = imread(
    "./../../data/images/DEHYDRATION_small_leaf_4_time_1_ax1cros_0950_Label_1152x1350_uint8.png", 0)

# Transform image with label to gmap
gmap = LabelMap.from_labels(image)

# Apply the wave propagation algorithm in order to get the distance transform
# from stomata to the cells of the leaf propagated in the air
generalized_wave_propagation_gmap(
    gmap,
    seed_labels=[labels['stomata']],
    propagation_labels=[labels['air']],
    target_labels=[labels['cell']]
)

dt_image = build_dt_grey_image_from_gmap(
    gmap,
    propagation_labels=[labels['stomata'], labels['air']],
    interpolate_missing_values=False
)

plot_dt_image(dt_image)
