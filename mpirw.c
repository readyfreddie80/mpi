#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

#define MASTER 0

typedef enum {
    LEFT,
    RIGHT,
    UP,
    DOWN
} direction_t;

omp_lock_t lock;
int steps = 10;


typedef struct particle_ctx_t {
	int x;
	int y;
	int n;
	int seed;
	int init_node;
} particle_ctx_t;

void push(particle_ctx_t **array, int *n, int *capacity, particle_ctx_t *element) {
    if(*n >= *capacity) {
        *capacity *= 2;
        *array = (particle_ctx_t*)realloc(*array, *capacity * sizeof(particle_ctx_t));
    }
    (*array)[*n] = *element;
    (*n)++;
}

void pop(particle_ctx_t **array, int *n, int index) {
    (*array)[index] = (*array)[(*n) - 1];
    (*n)--;
}

direction_t get_direction(int seed, double p_l,double p_r, double p_u, double p_d){
    double left = rand_r(&seed) * p_l;
    double right = rand_r(&seed) * p_r;
    double up = rand_r(&seed) * p_u;
    double down = rand_r(&seed) * p_d;

    if(left >= right && left >= up && left >= down) {
        return LEFT;
    }

    if(right >= left && right >= up && right >= down) {
        return RIGHT;
    }

    if(up >= right && up >= left && up >= down) {
        return UP;
    }

    return DOWN;
}

void random_walk(int rank,
                 int size,
                 int l,
                 int a,
                 int b,
                 int n,
                 int N,
                 double p_l,
                 double p_r,
                 double p_u,
                 double p_d) {

    omp_init_lock(&lock);

    int is_stop = 0;

    int particles_cnt = N;
    int particles_capacity = N;
    particle_ctx_t *particles = (particle_ctx_t*)malloc(N * sizeof(particle_ctx_t));

    int send_to_cnt[4] = {0};
    int send_to_capacity[4] = {N};
    particle_ctx_t *send_to[4];
    for(int i = 0; i < 4; i++){
        send_to[i] = (particle_ctx_t*)malloc(N * sizeof(particle_ctx_t));
    }

    int receive_from_capacity[4] = {0};
    particle_ctx_t *receive_from[4];
    int receive_tags[4] = {RIGHT, LEFT, DOWN, UP};

    //neighbors rank
    int neighbor[4];
    neighbor[LEFT] = rank - 1;
    neighbor[RIGHT] = rank + 1;
    neighbor[UP] = rank - a;
    neighbor[DOWN] = rank + a;


    int x = rank % a;
    int y = rank / a;
    //if on boarder
    //neighbor_rank = y_n * a + x_n
    if(x == 0)
        neighbor[LEFT] = rank + (a - 1);
    if(x == a - 1)
        neighbor[RIGHT] = a * y;
    if(y == 0)
        neighbor[UP] = (size - a) + x;
    if(y == b - 1)
        neighbor[DOWN] = x;

    int stopped_cnt = 0;
    int stopped_capacity = N;
    particle_ctx_t *stopped = (particle_ctx_t*) malloc(N * sizeof(particle_ctx_t));
    int *all_nodes = (int*)malloc(sizeof(int) * size);

    omp_set_lock(&lock);

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            #pragma omp task
            {
                while (!is_stop) {
                    omp_set_lock(&lock);
                    int i = 0;

                    while(i < particles_cnt) {
                        for(int j = 0; j < steps; j++) {
                            if(particles[i].n == 0) {
                                push_back(&stopped, &stopped_cnt, &stopped_capacity, particles + i);
                                pop(&particles, &particles_cnt, i);
                                i--;
                                break;
                            }
                            direction_t dir = get_direction(particles[i].seed, p_l, p_r, p_u, p_d);
                            if(dir == LEFT) {
                                particles[i].x -= 1;
                                if(particles[i].x < 0) {
                                    particles[i].x = l - 1;
                                    push(&send_to[LEFT], &send_to_cnt[LEFT], &send_to_capacity[LEFT], particles + i);
                                    pop(&particles, &particles_cnt, i);
                                    i-= 1;
                                    break;
                                }
                            }
                            if(dir == RIGHT) {
                                particles[i].x += 1;
                                if (particles[i].x >= l) {
                                    particles[i].x = 0;
                                    push(&send_to[RIGHT], &send_to_cnt[RIGHT], &send_to_capacity[RIGHT], particles + i);
                                    pop(&particles, &particles_cnt, i);
                                    i -= 1;
                                    break;

                                }
                            }
                            if(dir == UP) {
                                particles[i].y -= 1;
                                if (particles[i].y < 0) {
                                    particles[i].y = l - 1;
                                    push(&send_to[UP], &send_to_cnt[UP], &send_to_capacity[UP], particles + i);
                                    pop(&particles, &particles_cnt, i);
                                    i -= 1;
                                    break;
                                }
                            }
                            if(dir == DOWN) {
                                particles[i].y += 1;
                                if (particles[i].y >= l) {
                                    particles[i].y = 0;
                                    push(&send_to[DOWN], &send_to_cnt[DOWN], &send_to_capacity[DOWN], particles + i);
                                    pop(&particles, &particles_cnt, i);
                                    i -= 1;
                                    break;

                                }
                            }
                        }
                        i += 1;
                    }
                    omp_unset_lock(&lock);
                }
            }
            #pragma omp task
            {
                int *seeds = (int*)malloc(size * sizeof(int));
                if (rank == MASTER){
                      srand(time(NULL));
                      for (int i = 0; i < size; i++)
                        seeds[i] = (int)rand();
                }
                int seed;
                MPI_Scatter(seeds, 1, MPI_INT, &seed, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
                free(seeds);

                srand(seed);
                for(int i = 0; i < N; i++) {
                    particles[i].x = rand() % l;
                    particles[i].y = rand() % l;
                    particles[i].n = n;
                    particles[i].seed = (unsigned int)rand();
                    particles[i].init_node = rank;
                }

                omp_unset_lock(&lock);

                while (!is_stop) {

                    omp_set_lock(&lock);
                    MPI_Request requests[8];

                    for(int i = 0; i < 4; i++)
                        MPI_Isend(&send_to_cnt[i], 1, MPI_INT, neighbor[i], i, MPI_COMM_WORLD, requests + i);
                    for(int i = 0; i < 4; i++)
                        MPI_Irecv(&receive_from_capacity[i], 1, MPI_INT, neighbor[i], receive_tags[i], MPI_COMM_WORLD, requests + 4 + i);
                    MPI_Waitall(8, requests, MPI_STATUS_IGNORE);

                    for(int i = 0; i < 4; i++)
                        receive_from[i] = (particle_ctx_t*)malloc(receive_from_capacity[i] * sizeof(particle_ctx_t));
                    for(int i = 0; i < 4; i++)
                        MPI_Issend(send_to[i], sizeof(particle_ctx_t) * send_to_cnt[i], MPI_BYTE, neighbor[i], i, MPI_COMM_WORLD, requests + i)
                    for(int i = 0; i < 4; i++)
                         MPI_Irecv(receive_from[i], sizeof(particle_ctx_t) * receive_from_capacity[i], MPI_BYTE, neighbor[i], receive_tags[i], MPI_COMM_WORLD, requests + 4 + i);
                    MPI_Waitall(8, requests, MPI_STATUS_IGNORE);

                    for(int i = 0; i < 4; i++){
                        for (int j = 0; j < receive_from_capacity[i]; j++) {
                            push(&particles, &particles_cnt, &particles_capacity, receive_from[i] + j);
                        }
                        free(receive_from[i]);
                        send_to_cnt[i] = 0;
                    }

                    int all_stopped_cnt;
                    MPI_Reduce(&stopped_cnt, &all_stopped_cnt, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);

                    MPI_Barrier(MPI_COMM_WORLD);

                    if (rank == MASTER) {
                        if (all_stopped_cnt == size * N) {
                        is_stop = 1;
                        }
                    }
                    MPI_Bcast(&is_stop, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

                    if (is_stop) {
                        MPI_Gather(&stopped_cnt, 1, MPI_INT, all_nodes, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
                    }
                    omp_unset_lock(&lock);
                }

                MPI_Barrier(MPI_COMM_WORLD);

                MPI_File data;
                MPI_File_delete("data.bin", MPI_INFO_NULL);
                MPI_File_open(MPI_COMM_WORLD, "data.bin", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &data);

                int a_tbl = rank % a;
                int b_tbl= rank / a;
                int y_size = l;
                int x_size = l * size;

                int **result = calloc(y_size, sizeof(int*));

                for(int i = 0; i < y_size; i++) {
                    result[i] = calloc(x_size, sizeof(int));
                }

                for (int i = 0; i < stopped_cnt; i++) {
                    int y = stopped[i].y;
                    int x = stopped[i].x;
                    int init_node = stopped[i].init_node;
                    result[y][x * size + init_node] += 1;
                }

                for (int y_i = 0; y_i < y_size; y_i++) {
                    for (int x_i = 0; x_i < l; x_i++) {
                        int bytes_line = l * size * a * sizeof(int);
                        int bytes_y_i = bytes_line * b_tbl * l + bytes_line * y_i;
                        int bytes_x_i = bytes_y_i + a_tbl * l * size * sizeof(int) + x_i * size * sizeof(int);
                        MPI_File_set_view(data, bytes_x_i, MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
                        MPI_File_write(data, &result[y_i][x_i * size], size, MPI_INT, MPI_STATUS_IGNORE);
                    }
                }

                MPI_File_close(&data);

                for (int i = 0; i < y_size; i++) {
                    free(result[i]);
                }
                free(result);
            }
        }
    }
    #pragma omp taskwait
        omp_destroy_lock(&lock);

    free(all_nodes);
    free(stopped);
    free(particles);
    for(int i = 0; i < 4; i++) {
        free(send_to[i])
    }
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int l = atoi(argv[1]);
    int a = atoi(argv[2]);
    int b = atoi(argv[3]);
    int n = atoi(argv[4]);
    int N = atoi(argv[5]);
    double p_l = atof(argv[6]);
    double p_r = atof(argv[7]);
    double p_u = atof(argv[8]);
    double p_d = atof(argv[9]);


    int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    omp_set_num_threads(2);

    double start, end;
    if(rank == MASTER) {
        start =  MPI_Wtime();
    }

    random_walk(rank, size, l, a, b, n, N, p_l, p_r, p_u, p_d);

    if(rank == MASTER) {
        end = MPI_Wtime();
        double delta = end - start;
        FILE *f = fopen("stats.txt", "w");
        fprintf(f, "%d %d %d %d %d %f %f %f %f, %fs", l, a, b, n, N, p_l, p_r, p_u, p_d, delta);

        fclose(f);
    }
    MPI_Finalize();
    return 0;
}
