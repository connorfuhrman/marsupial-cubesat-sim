#include <stdio.h>
#include <stdlib.h>


typedef struct {
  FILE *datafile;  
  unsigned int num_datapoints;
  double **data;
} particleData_t;


void open_data_file(particleData_t *data, char *file) {  
  data->datafile = fopen(file, "r");
}

void read_data_file(particleData_t *data) {
  int 
}
