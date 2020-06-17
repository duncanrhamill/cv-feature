'''
Generate test vectors for the FAST algorithm.

Run from the test_vectors directory.
'''

import cv2
import numpy as np
import csv

# Load the westminster image
img = cv2.imread('../res/westminster.jpg')

# Convert to grey
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Create and run FAST
fast = cv2.FastFeatureDetector_create(threshold=51, nonmaxSuppression=0, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
keypoints = fast.detect(img, None)

print(f'Found {len(keypoints)} keypoints')

kps = [kp.pt for kp in keypoints]

with open('fast_test_vectors.csv', 'w') as f:
    writer = csv.writer(f)
    for kp in kps:
        writer.writerow(kp)
