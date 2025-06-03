import numpy as np
import matplotlib.pyplot as plt

# Fungsi yang akan dioptimasi
def objective_function(x):
    return x**2

# Parameter PSO
num_particles = 10
max_iterations = 50
w = 0.5  # Inersia
c1 = 1.5 # Koefisien kognitif
c2 = 1.5 # Koefisien sosial
search_space_min = -10
search_space_max = 10

# Inisialisasi partikel
particles_position = np.random.uniform(search_space_min, search_space_max, num_particles)
particles_velocity = np.random.uniform(-(search_space_max - search_space_min) * 0.1, (search_space_max - search_space_min) * 0.1, num_particles) # Kecepatan awal yang lebih kecil

# Inisialisasi pBest (personal best) untuk setiap partikel
pBest_position = np.copy(particles_position)
pBest_value = np.array([objective_function(p) for p in pBest_position])

# Inisialisasi gBest (global best)
gBest_value = np.min(pBest_value)
gBest_position = pBest_position[np.argmin(pBest_value)]

# Variabel untuk menyimpan histori gBest_value untuk plotting
gBest_values_per_iteration = [] # Inisialisasi sebagai list kosong

# Iterasi Utama PSO
for iteration in range(max_iterations):
    for i in range(num_particles):
        # Update kecepatan partikel
        r1 = np.random.rand()
        r2 = np.random.rand()
        particles_velocity[i] = (w * particles_velocity[i] +
                                 c1 * r1 * (pBest_position[i] - particles_position[i]) +
                                 c2 * r2 * (gBest_position - particles_position[i]))

        # Update posisi partikel
        particles_position[i] = particles_position[i] + particles_velocity[i]

        # Batasi partikel dalam ruang pencarian
        if particles_position[i] < search_space_min:
            particles_position[i] = search_space_min
            # particles_velocity[i] = 0 # Opsional: reset kecepatan jika menabrak batas
        elif particles_position[i] > search_space_max:
            particles_position[i] = search_space_max
            # particles_velocity[i] = 0 # Opsional: reset kecepatan jika menabrak batas

        # Evaluasi fitness partikel
        current_fitness = objective_function(particles_position[i])

        # Update pBest
        if current_fitness < pBest_value[i]:
            pBest_value[i] = current_fitness
            pBest_position[i] = particles_position[i]

    # Update gBest
    current_min_pBest_value = np.min(pBest_value)
    if current_min_pBest_value < gBest_value:
        gBest_value = current_min_pBest_value
        gBest_position = pBest_position[np.argmin(pBest_value)]

    # Menyimpan histori gBest_value untuk plotting
    gBest_values_per_iteration.append(gBest_value)

# Hasil akhir
print("\nOptimasi Selesai.")
print(f"Nilai minimum yang ditemukan (gBest Value): {gBest_value:.6f}")
print(f"Posisi x terbaik (gBest Position): {gBest_position:.6f}")

# Membuat grafik nilai terbaik per iterasi
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_iterations + 1), gBest_values_per_iteration, marker='o', linestyle='-')
plt.title('Nilai Terbaik Global (gBest Value) per Iterasi')
plt.xlabel('Iterasi')
plt.ylabel('gBest Value (f(x))')
plt.grid(True)
plt.show()