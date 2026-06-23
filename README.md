# Elevator-Fullness-Detection


This project uses OpenCV to detect the object contours inside a box, which simulates the inside space of an elevator.

## Method

The system calculates the occupancy rate by dividing the detected object contour area by the total box area.

If the occupancy rate is greater than or equal to 80%, the system judges the elevator as full.

## Run

python main.py

## Result

Occupancy: 58.7%
Status: NOT FULL
