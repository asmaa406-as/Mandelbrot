#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

int mandelbrot(double c_real, double c_imag, int max_iter)
{
    double real = 0;
    double imag = 0;

    for (int i = 1; i <= max_iter; i++)
    {
        double new_real = (real * real - imag * imag) + c_real;
        double new_imag = (2 * real * imag) + c_imag;

        real = new_real;
        imag = new_imag;

        double magnitude_squared = real * real + imag * imag;
        if (magnitude_squared > 4)
        {
            return i;
        }
    }

    return 0;
}

int main()
{
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int width, height, max_iter;

    if (rank == 0)
    {
        printf("Enter image width: ");
        fflush(stdout);
        scanf("%d", &width);

        printf("Enter image height: ");
        fflush(stdout);
        scanf("%d", &height);

        printf("Enter maximum iterations: ");
        fflush(stdout);
        scanf("%d", &max_iter);
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (width < 2 || height < 2 || max_iter < 1)
    {
        if (rank == 0)
        {
            printf("Width and height must be at least 2 and maximum iterations must be at least 1.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int base_rows = height / size;
    int remainder = height % size;
    int local_rows = base_rows + (rank < remainder ? 1 : 0);

    int start_row = rank * base_rows + (rank < remainder ? rank : remainder);
    int end_row = start_row + local_rows;

    int local_count = local_rows * width * 3;
    int *local_pixels = malloc((local_count > 0 ? local_count : 1) * sizeof(int));

    if (local_pixels == NULL)
    {
        printf("Memory allocation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *all_pixels = NULL;
    int *recv_counts = NULL;
    int *start_positions = NULL;

    if (rank == 0)
    {
        all_pixels = malloc((size_t)width * height * 3 * sizeof(int));
        recv_counts = malloc(size * sizeof(int));
        start_positions = malloc(size * sizeof(int));

        if (all_pixels == NULL || recv_counts == NULL || start_positions == NULL)
        {
            printf("Memory allocation failed!\n");
            free(local_pixels);
            free(all_pixels);
            free(recv_counts);
            free(start_positions);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            int rows = base_rows + (i < remainder ? 1 : 0);
            recv_counts[i] = rows * width * 3;
            start_positions[i] = offset;
            offset += recv_counts[i];
        }
    }

    for (int y = start_row; y < end_row; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double c_real = -2.0 + (3.0 * x / (width - 1));
            double c_imag = -1.5 + (3.0 * y / (height - 1));

            int iterations = mandelbrot(c_real, c_imag, max_iter);

            int red, green, blue;

            if (iterations == 0)
            {
                red = 0;
                green = 0;
                blue = 0;
            }
            else
            {
                double t = (double)iterations;

                red   = (int)(127.5 * (1 + sin(0.5 * t)));
                green = (int)(127.5 * (1 + sin(0.5 * t + 2.0)));
                blue  = (int)(127.5 * (1 + sin(0.5 * t + 4.0)));
            }

            int pixel_index = ((y - start_row) * width + x) * 3;
            local_pixels[pixel_index] = red;
            local_pixels[pixel_index + 1] = green;
            local_pixels[pixel_index + 2] = blue;
        }
    }

    MPI_Gatherv(
        local_pixels,
        local_count,
        MPI_INT,
        all_pixels,
        recv_counts,
        start_positions,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        FILE *image = fopen("mandelbrot.ppm", "w");

        if (image == NULL)
        {
            printf("Error creating file!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fprintf(image, "P3\n%d %d\n255\n", width, height);

        for (int i = 0; i < width * height * 3; i += 3)
        {
            fprintf(image, "%d %d %d\n",
                    all_pixels[i],
                    all_pixels[i + 1],
                    all_pixels[i + 2]);
        }

        fclose(image);
        printf("Mandelbrot image generated as mandelbrot.ppm\n");
    }

    free(local_pixels);

    if (rank == 0)
    {
        free(all_pixels);
        free(recv_counts);
        free(start_positions);
    }

    MPI_Finalize();
    return 0;
}