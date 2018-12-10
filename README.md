# CrIS_VIIRS_collocation

Fast and Accurate collocation of Fast and Accurate Collocation of the Visible Infrared Imaging Radiometer Suite (VIIRS) Measurements with Cross-Track Infrared Sounder (CrIS) based the method from the paper: 

Wang, L.; Tremblay, D.; Zhang, B.; Han, Y. Fast and Accurate Collocation of the Visible Infrared Imaging Radiometer Suite Measurements with Cross-Track Infrared Sounder. Remote Sens. 2016, 8, 76.

The paper can be assessed at, 

https://www.mdpi.com/2072-4292/8/1/76/htm

Noted that the equation (4) have a typo for the term y0*sin(lm)*sin(ph) ,which should be -y0*sin(lm)*sin(ph). 

===================================================
1) check "code_test.py" for the driver, the package needs pykdtree from https://github.com/storpipfugl/pykdtree for fast searching. Other packages can be easily found from Anaconda distribution of Python 2.7 . 

2) As you may find, making plots is really time consuming. You can comment this part off.  


