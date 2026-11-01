"""Galaxy10 DECaLS classes (Galaxy Zoo volunteer labels) and how they map to the app's labels."""

CLASSES = [
    "Disturbed", "Merging", "Round smooth", "In-between round smooth", "Cigar-shaped smooth",
    "Barred spiral", "Unbarred tight spiral", "Unbarred loose spiral", "Edge-on without bulge", "Edge-on with bulge",
]

# Coarse groups = the buttons in Inspect mode (and the 3D model shapes).
COARSE = {
    "merger": [0, 1],
    "elliptical": [2, 3, 4],
    "spiral": [5, 6, 7],
    "edge-on": [8, 9],
}
COARSE_OF = {i: k for k, idx in COARSE.items() for i in idx}

# Active learning adds an 11th class the Galaxy10 data never had: stars, nebula gas, artifacts, planets.
NOT_GALAXY = "Not a galaxy"
CLASSES_ALL = CLASSES + [NOT_GALAXY]
# what people can answer in the "teach me" queue -> (fine class index or None, coarse group or None)
TEACH_LABELS = {
    "spiral": (None, "spiral"), "elliptical": (None, "elliptical"), "edge-on": (None, "edge-on"), "merger": (None, "merger"),
    "star": (10, None), "gas": (10, None), "artifact": (10, None),
}
