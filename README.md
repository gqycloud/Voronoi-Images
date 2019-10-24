# Voronoi-Images
Python script for processing voronoi diagrams out of images (voronoi.py)
Processes images as follows: 

1) Read Image
2) Use various feature detection algorithms from the OpenCV library to process feature points
3) Generate voronoi facets from feature points
4) Average out the color within each feature facet to create abstract tesselations

Example: 

TestImage: 

![testImage](/testImage.jpg)

SURF feature detection: 

![SURF](/SURF.jpg)

FAST feature detection: 

![FAST](/FAST.jpg)

BRIEF feature detection: 

![BRIEF](/BRIEF.jpg)

ORB feature detection: 

![ORB](/ORB.jpg)
