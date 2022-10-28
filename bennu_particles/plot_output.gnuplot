set datafile separator ','
# set autoscale fix
set xrange [-5:5]
set yrange [-5:5]
set zrange [-5:5]

# Remote the z axis offset
set xyplane 0

FILES = system("ls run*.csv")

set xlabel "x [km]"
set ylabel "y [km]"
set xlabel "x [km]"
splot for [f in FILES] f u ($1/1000):($2/1000):($3/1000) w lines linetype rgb 'blue' notitle