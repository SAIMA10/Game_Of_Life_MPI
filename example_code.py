import sys, time  # imports sys and time libraries which allows us to use input parameters and sleep.
from math import pi  # imports Pi constant from the library math.
import numpy as np

time.sleep(60)  # sleeps for 60 seconds.
print("Radius is = %s" % sys.argv[1])  # prints on screen the input parameter.
radius = float(
    sys.argv[1]
)  # transforms the parameter from text to double precision number.
circumference = 2 * pi * radius  # computes the length of the circumference.
print(
    "Circumference is = %.5f" % circumference
)  # prints on screen the value of the circumference.
area = pi * pow(radius, 2)  # computes the area of the circle.
print("Area is = %.5f" % area)  # prints on screen the value of the area.
