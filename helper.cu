#include <math.h>
#include <fstream>
#include <iostream>

#include "helper.h"

using namespace std;

#define MAX_ELEM 1000000

/*
 * geoDistance_kernel computes geographical distance (lat1, lat1) and (lat2, lon2)
 * It is the kernel function that gets called
 */
__global__ void geoDistance_kernel(float *lat, float *lon, const size_t size, int *pop, int *res, float kmRange)
{
    /* get the index of the city for which we compute the accessible population */
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    /* if index exceeds the size of the array, then skip computation */
    if (i >= size) {
            return;
    }

    /* initialize vector with 0 */
    res[i] = 0;

    /* loop through all the cities, first initialize some data needed */
    float register lat1 = (90 - lat[i]) * DEGREE_TO_RADIANS;
    float register lon1 = lon[i] * DEGREE_TO_RADIANS;
    float register radius = 6371 * 6371;
    float register kmRange_pythagoras = kmRange * kmRange;
    float register lat2, lon2, distance_x, distance_y, distance;
    for (int j = 0; j < size; j += 3) {
        /* take the latitude */
        lat2 = (90 - lat[j]) * DEGREE_TO_RADIANS;

        /* take the longitude */
        lon2 = lon[j] * DEGREE_TO_RADIANS;

        /* use Pythagora to compute the two sides of the triangle */
        distance_x = (lat2 - lat1) * (lat2 - lat1);
        distance_y = (lon2 - lon1) * (lon2 - lon1);

        /* get the desired side of the triangle */
        distance = distance_x + distance_y;

        /* check kmRange with the actual distance */
        if (distance * radius <= kmRange_pythagoras)
                res[i] += pop[j];

        /* perform second step of loop unrolling */
        lat2 = (90 - lat[j + 1]) * DEGREE_TO_RADIANS;
        lon2 = lon[j + 1] * DEGREE_TO_RADIANS;
        distance_x = (lat2 - lat1) * (lat2 - lat1);
        distance_y = (lon2 - lon1) * (lon2 - lon1);
        distance = distance_x + distance_y;
        if (distance * radius <= kmRange_pythagoras)
                res[i] += pop[j + 1];

        /* perform third step of loop unrolling */
        lat2 = (90 - lat[j + 2]) * DEGREE_TO_RADIANS;
        lon2 = lon[j + 2] * DEGREE_TO_RADIANS;
        distance_x = (lat2 - lat1) * (lat2 - lat1);
        distance_y = (lon2 - lon1) * (lon2 - lon1);
        distance = distance_x + distance_y;
        if (distance * radius <= kmRange_pythagoras)
                res[i] += pop[j + 2];
    }
}

// sampleFileIO demos reading test files and writing output
void sampleFileIO(float kmRange, const char* fileIn, const char* fileOut)
{
    /* declare the needed host buffers */
    string register geon;
    float register *lat = (float *) malloc(MAX_ELEM * sizeof(float));
    float register *lon = (float *) malloc(MAX_ELEM * sizeof(float));
    int register *pop = (int *) malloc(MAX_ELEM * sizeof(int));

    int register *res = (int *) malloc(MAX_ELEM * sizeof(int));

    /* declare and initialize file descriptors */
    ifstream ifs(fileIn);
    ofstream ofs(fileOut);

    /* declare and alloc device buffers */
    float register *lat_device = 0;
    float register *lon_device = 0;
    int register *res_device = 0;
    int register *pop_device = 0;

    cudaMalloc((void **) &lat_device, MAX_ELEM * sizeof(float));
    cudaMalloc((void **) &lon_device, MAX_ELEM * sizeof(float));
    cudaMalloc((void **) &res_device, MAX_ELEM * sizeof(int));
    cudaMalloc((void **) &pop_device, MAX_ELEM * sizeof(int));

    /* read from file */
    int it = 0;

    while(ifs >> geon >> lat[it] >> lon[it] >> pop[it])
        it++;

    /* copy data to device */
    cudaMemcpy(lat_device, lat, MAX_ELEM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(lon_device, lon, MAX_ELEM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(res_device, res, MAX_ELEM * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pop_device, pop, MAX_ELEM * sizeof(int), cudaMemcpyHostToDevice);

    /* declare block size and number of blocks */
    const size_t register block_size = 256;
    size_t register no_blocks = it / block_size;

    if (it % block_size != 0)
        no_blocks += 1;

    /* call the kernel function */
    geoDistance_kernel<<<no_blocks, block_size>>>(lat_device, lon_device, it, pop_device, res_device, kmRange);

    /* synchronize devices */
    cudaDeviceSynchronize();

    /* copy results from device to host */
    cudaMemcpy(res, res_device, MAX_ELEM * sizeof(int), cudaMemcpyDeviceToHost);

    /* print results to file */
    for (int i = 0; i < it; i++) {
        ofs << res[i] << endl;
    }

    /* free allocated memory */
    cudaFree(lat_device);
    cudaFree(lon_device);
    cudaFree(res_device);
    cudaFree(pop_device);
    free(lat);
    free(lon);
    free(res);
    free(pop);
    ifs.close();
    ofs.close();
}