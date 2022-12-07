set datafile separator ','
# set autoscale fix
RANGE_KM = 1
set xrange [-1*RANGE_KM:RANGE_KM]
set yrange [-1*RANGE_KM:RANGE_KM]
set zrange [-1*RANGE_KM:RANGE_KM]

# Remote the z axis offset
set xyplane 0

FILES = system("ls run*.csv")

set xlabel "x [km]"
set ylabel "y [km]"
set xlabel "x [km]"
splot for [f in FILES] f u ($2/1000):($3/1000):($4/1000) w lines linetype rgb 'blue' notitle
