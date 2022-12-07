set print "-"
set datafile separator ','
set term pngcairo enhanced font "Times New Roman,12.0" size 1500,1100

# set autoscale fix
RANGE_KM = 1
set xrange [-1*RANGE_KM:RANGE_KM]
set yrange [-1*RANGE_KM:RANGE_KM]
set zrange [-1*RANGE_KM:RANGE_KM]

# Remote the z axis offset
set xyplane 0

NFILES = int(system("ls *.csv | wc -l"))  # How many files to plot
array FILES[NFILES]  # File names
array NDATAPOINTS[NFILES]  # Array to store the # of datapoints
MAX_NDATAPOINTS = int(0.0)

# Count all the entries in each file
do for [i = 1:NFILES] {
  FILES[i] = sprintf("run_%d.csv", i)
  NDATAPOINTS[i] = int(system(sprintf("cat %s | wc -l", FILES[i])))
  if (NDATAPOINTS[i] > MAX_NDATAPOINTS) {
     MAX_NDATAPOINTS = NDATAPOINTS[i]
  }
}

print sprintf("Got %d files", NFILES)
print sprintf("Animating %d frames", MAX_NDATAPOINTS)


set xlabel "x [km]"
set ylabel "y [km]"
set xlabel "x [km]"


# splot for [f in FILES] f u ($2/1000):($3/1000):($4/1000) w lines linetype rgb 'blue' notitle
time = 0.0
dt = 5
do for [t = 1:MAX_NDATAPOINTS] {
  set output sprintf("/tmp/bennu_particle_ejection_frames/frame_%d.png", t)
  set title sprintf("Bennu Particle Ejection Simulation\nTime %.2f - Frame %d of %d", time, t, MAX_NDATAPOINTS)
  set multiplot
  #print sprintf("Plotting frame %d...", t)
  do for [i = 1:NFILES] {
    if (NDATAPOINTS[i] < t) {
        splot FILES[i] every ::0::t u ($2/1000):($3/1000):($4/1000) w lines linetype rgb 'blue' notitle
    }
    else {
        splot FILES[i] every ::0::NDATAPOINTS[i] u ($2/1000):($3/1000):($4/1000) w lines linetype rgb 'blue' notitle
    }
  }

  # if (t == 1) {
  #   unset border
  #   unset xtics
  #   unset ytics
  # }
  unset multiplot
  time = time + dt
}
