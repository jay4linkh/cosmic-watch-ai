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
