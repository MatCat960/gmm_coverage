#!/bin/bash

x1=$((RANDOM % 251))
y1=$((RANDOM % 175))
x2=$((RANDOM % 251))
y2=$((RANDOM % 175))
x3=$((RANDOM % 251))
y3=$((RANDOM % 175))
x4=$((RANDOM % 251))
y4=$((RANDOM % 175))
x5=$((RANDOM % 251))
y5=$((RANDOM % 175))
x6=$((RANDOM % 251))
y6=$((RANDOM % 175))
x7=$((RANDOM % 251))
y7=$((RANDOM % 175))
x8=$((RANDOM % 251))
y8=$((RANDOM % 175))
x9=$((RANDOM % 251))
y9=$((RANDOM % 175))
x10=$((RANDOM % 251))
y10=$((RANDOM % 175))
x11=$((RANDOM % 251))
y11=$((RANDOM % 175))
x12=$((RANDOM % 251))
y12=$((RANDOM % 175))

roslaunch gmm_coverage flightmare_gmm.launch x1:=$x1 y1:=$y1 x2:=$x2 y2:=$y2 x3:=$x3 y3:=$y3 x4:=$x4 y4:=$y4 x5:=$x5 y5:=$y5 x6:=$x6 y6:=$y6 x7:=$x7 y7:=$y7 x8:=$x8 y8:=$y8 x9:=$x9 y9:=$y9 x10:=$x10 y10:=$y10 x11:=$x11 y11:=$y11 x12:=$x12 y12:=$y12