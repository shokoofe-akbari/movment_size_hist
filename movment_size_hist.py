import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.cluster import KMeans
import os
import datetime
import shutil

time = r"\2024-06-29_12-55-48"


def create_analysis_folder(base_directory='D:/PhD thesis/Data Analysis Code-Python'):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    analysis_folder_path = os.path.join(base_directory, f'Analysis_{timestamp}')
    os.makedirs(analysis_folder_path, exist_ok=True)
    return analysis_folder_path


def copy_input_files(input_files, destination_folder):
    for file_path in input_files:
        shutil.copy(file_path, destination_folder)


def save_plot(figure, filename, output_folder):
    file_path = os.path.join(output_folder, f'{filename}.pdf')
    figure.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.close(figure)


def write_summary_file(destination_folder, content):
    summary_file_path = os.path.join(destination_folder, 'analysis_summary.txt')
    with open(summary_file_path, 'w') as file:
        file.write(content)


def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    step_data = []

    for line in lines:
        parts = line.split()

        if len(parts) == 1 and parts[0].isdigit():
            if step_data:
                data.append(step_data)
            step_data = []

        elif len(parts) >= 3:
            try:
                x, y, z = map(float, parts[:3])
                step_data.append((x, y, z))
            except ValueError:
                continue

    if step_data:
        data.append(step_data)

    return data


def calculate_step_sizes(data):
    step_sizes = []
    num_steps = len(data)
    num_atoms = len(data[0])

    for step in range(1, num_steps):
        for atom in range(num_atoms):
            prev_position = np.array(data[step - 1][atom][1:4])
            current_position = np.array(data[step][atom][1:4])
            step_size = np.linalg.norm(current_position - prev_position)
            step_sizes.append(step_size)

    return step_sizes


def calculate_step_sizes_for_atom(data, atom_id):
    step_sizes = []
    num_steps = len(data)

    for step in range(1, num_steps):
        prev_position = None
        current_position = None

        for atom in data[step - 1]:
            if atom[0] == atom_id:
                prev_position = np.array(atom[1:4])
                break

        for atom in data[step]:
            if atom[0] == atom_id:
                current_position = np.array(atom[1:4])
                break

        if prev_position is not None and current_position is not None:
            step_size = np.linalg.norm(current_position - prev_position)
            step_sizes.append(step_size)

    return step_sizes


def plot_histogram(step_sizes, title):
    plt.figure(figsize=(10, 6))
    plt.hist(step_sizes, bins=30, edgecolor='black')
    plt.title(title)
    plt.xlabel('Step Size')
    plt.ylabel('Frequency')
    plt.show()


# Plot histogram for step sizes at a specific time step
def plot_histogram_for_time_step(data, time_step):
    step_sizes = []
    if time_step < 1 or time_step >= len(data):
        raise ValueError("Invalid time step. Please select a time step between 1 and the total number of steps.")

    num_atoms = len(data[0])
    for atom in range(num_atoms):
        prev_position = np.array(data[time_step - 1][atom][1:4])
        current_position = np.array(data[time_step][atom][1:4])
        step_size = np.linalg.norm(current_position - prev_position)
        step_sizes.append(step_size)

    plot_histogram(step_sizes, f'Histogram of Step Sizes at Time Step {time_step}')


# Plot histogram for a specific atom over all time steps
def plot_histogram_for_atom(data, atom_id):
    step_sizes = calculate_step_sizes_for_atom(data, atom_id)
    plot_histogram(step_sizes, f'Histogram of Step Sizes for Atom {atom_id} over All Time Steps')


def calculate_density(data, grid_size):
    all_coords = []
    for step in data:
        for atom in step:
            if len(atom) == 3:  # Ensure the atom tuple has 3 elements
                all_coords.append((atom[0], atom[1], atom[2]))
            else:
                print(f"Unexpected atom structure: {atom}")

    all_coords = np.array(all_coords)

    if all_coords.size == 0:
        raise ValueError("No valid coordinates found in data.")

    # Extent of the coordinate data
    min_coord = np.floor(all_coords.min(axis=0) / grid_size) * grid_size
    max_coord = np.ceil(all_coords.max(axis=0) / grid_size) * grid_size
    dims = ((max_coord - min_coord) / grid_size).astype(int) + 1

    # Create an empty density map
    density_map = np.zeros(dims)

    # Populate the density map
    for step in data:
        for atom in step:
            if len(atom) == 3:  # Ensure the atom tuple has 3 elements
                indices = ((np.array(atom) - min_coord) / grid_size).astype(int)
                density_map[tuple(indices)] += 1

    return density_map


def plot_density_map(density_map, grid_size):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get indices where density is non-zero
    non_zero_indices = np.argwhere(density_map > 0)
    densities = density_map[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]]

    # Scale the sizes for visibility
    sizes = densities * 100

    ax.scatter(non_zero_indices[:, 0] * grid_size, non_zero_indices[:, 1] * grid_size,
               non_zero_indices[:, 2] * grid_size, s=sizes, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Density Map of Particles')
    plt.show()


def plot_trajectory(data, atom_id=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_steps = len(data)
    num_atoms = len(data[0])

    if atom_id is not None:
        # Plot trajectory for the specific atom
        trajectory = np.array([step[atom_id] for step in data])
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=f'Atom {atom_id} Trajectory')
    else:
        # Plot trajectories for all atoms
        for atom in range(num_atoms):
            trajectory = np.array([step[atom] for step in data])
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=f'Atom {atom} Trajectory')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Trajectory of Atoms Over Time')
    plt.legend()
    plt.show()


def calculate_displacement_statistics(data):
    num_steps = len(data)
    num_atoms = len(data[0])

    displacements = {atom: [] for atom in range(num_atoms)}
    speeds = {atom: [] for atom in range(num_atoms)}

    for atom in range(num_atoms):
        total_displacement = 0.0
        for step in range(1, num_steps):
            prev_position = np.array(data[step - 1][atom])
            current_position = np.array(data[step][atom])
            displacement = np.linalg.norm(current_position - prev_position)
            displacements[atom].append(displacement)
            speeds[atom].append(displacement)  # Assuming unit time between steps
            total_displacement += displacement

        displacements[atom] = np.array(displacements[atom])
        speeds[atom] = np.array(speeds[atom])

    statistics = {
        atom: {
            "mean_displacement": np.mean(displacements[atom]),
            "variance_displacement": np.var(displacements[atom]),
            "mean_speed": np.mean(speeds[atom]),
            "total_displacement": np.sum(displacements[atom])
        }
        for atom in range(num_atoms)
    }

    return statistics


def calculate_correlation(data1, data2):

    num_steps = min(len(data1), len(data2))  # Ensure we only compare up to the shortest dataset
    num_atoms = min(len(data1[0]), len(data2[0]))  # Ensure the number of atoms is consistent

    correlations = {}

    for atom in range(num_atoms):
        displacements1 = []
        displacements2 = []

        for step in range(1, num_steps):
            displacement1 = np.linalg.norm(np.array(data1[step][atom]) - np.array(data1[step - 1][atom]))
            displacement2 = np.linalg.norm(np.array(data2[step][atom]) - np.array(data2[step - 1][atom]))

            displacements1.append(displacement1)
            displacements2.append(displacement2)

        # Calculate Pearson correlation coefficient
        if len(displacements1) > 1:  # Correlation requires at least two data points
            correlation, _ = pearsonr(displacements1, displacements2)
            correlations[atom] = correlation
        else:
            correlations[atom] = None  # Not enough data to calculate correlation

    return correlations


def calculate_correlation_matrix(data):
    num_steps = len(data)
    num_atoms = len(data[0])

    # Initialize a matrix to hold the displacements
    displacements = np.zeros((num_atoms, num_steps - 1))

    # Calculate displacements for each atom across all time steps
    for atom in range(num_atoms):
        for step in range(1, num_steps):
            displacement = np.linalg.norm(np.array(data[step][atom]) - np.array(data[step - 1][atom]))
            displacements[atom, step - 1] = displacement

    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(displacements)

    return correlation_matrix


def plot_heatmap(correlation_matrix):

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Heatmap of Atom Displacements')
    plt.xlabel('Atom Index')
    plt.ylabel('Atom Index')
    plt.show()


def plot_heatmap_and_save(correlation_matrix, output_file):

    plt.figure(figsize=(20, 16))  # Increase the figure size for better visibility
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True,
                annot_kws={"size": 8})  # Adjust font size for better readability
    plt.title('Correlation Heatmap of Atom Displacements')
    plt.xlabel('Atom Index')
    plt.ylabel('Atom Index')

    plt.savefig(output_file, format='pdf', bbox_inches='tight')  # Save as PDF with tight bounding box
    plt.close()  # Close the figure to avoid display


def calculate_msd(data):

    num_steps = len(data)
    num_atoms = len(data[0])

    msd = np.zeros(num_steps)

    for step in range(num_steps):
        sum_displacement = 0.0
        for atom in range(num_atoms):
            initial_position = np.array(data[0][atom])
            current_position = np.array(data[step][atom])
            displacement = np.linalg.norm(current_position - initial_position) ** 2
            sum_displacement += displacement
        msd[step] = sum_displacement / num_atoms

    return msd


def cluster_particles(data, num_clusters=3):

    final_positions = np.array([data[-1][atom] for atom in range(len(data[0]))])
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(final_positions)
    labels = kmeans.labels_

    return labels


def calculate_angular_distribution(data):

    num_steps = len(data)
    num_atoms = len(data[0])

    angles = []

    for atom in range(num_atoms):
        for step in range(1, num_steps - 1):
            vec1 = np.array(data[step][atom]) - np.array(data[step - 1][atom])
            vec2 = np.array(data[step + 1][atom]) - np.array(data[step][atom])
            if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                angles.append(np.degrees(angle))

    return angles


def save_plot_as_pdf(figure, filename, output_folder):

    file_path = os.path.join(output_folder, f'{filename}.pdf')
    figure.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.close(figure)  # Close the figure to free up memory


"""
نتایجی که از محاسبات به دست آمده، نشان‌دهنده اطلاعات کلیدی در مورد حرکت ذرات (اتم‌ها) در طول زمان است. این اطلاعات به شما کمک می‌کنند تا تفاوت‌های بین سناریوهای مختلف شبیه‌سازی را تجزیه و تحلیل کنید. در اینجا توضیحی در مورد هر یک از این معیارها و نحوه استفاده از آنها برای گرفتن نتایج مختلف ارائه می‌شود:

### توضیحات معیارهای آماری:

1. **میانگین جابجایی (`mean_displacement`)**:
   - **توضیح**: این مقدار نشان می‌دهد که یک ذره به طور متوسط در هر گام زمانی چه مسافتی را جابجا می‌کند. این مقدار، شاخصی از پویایی یا فعالیت ذره است.
   - **تفسیر**: میانگین جابجایی بالاتر به این معنی است که ذره به طور متوسط در هر گام زمانی بیشتر حرکت می‌کند. برای مثال، میانگین جابجایی 0.854 برای ذره اول در مجموعه داده دوم به این معنی است که این ذره به طور متوسط بیشتر از ذره اول در مجموعه داده اول (0.354) جابجا می‌شود.

2. **واریانس جابجایی (`variance_displacement`)**:
   - **توضیح**: واریانس جابجایی نشان می‌دهد که تغییرات جابجایی‌ها در طول زمان چقدر متغیر است. به عبارت دیگر، آیا حرکت ذره به صورت پیوسته و یکنواخت است یا به صورت ناگهانی و غیرقابل پیش‌بینی؟
   - **تفسیر**: واریانس بالاتر به این معنی است که میزان جابجایی در گام‌های زمانی مختلف بسیار متغیر است. برای مثال، واریانس 0.210 برای ذره اول در مجموعه داده دوم نسبت به واریانس 0.021 در مجموعه داده اول نشان می‌دهد که جابجایی ذره در سناریوی دوم بسیار نامنظم‌تر است.

3. **سرعت متوسط (`mean_speed`)**:
   - **توضیح**: سرعت متوسط، معیاری از میزان جابجایی یک ذره در هر گام زمانی است و به طور مستقیم با میانگین جابجایی مرتبط است. در اینجا، سرعت و جابجایی به دلیل فرض واحد زمانی یکسان، برابر هستند.
   - **تفسیر**: این مقدار مانند میانگین جابجایی تفسیر می‌شود، اما به عنوان یک شاخص برای سرعت حرکت ذره در طول زمان.

4. **جابجایی کل (`total_displacement`)**:
   - **توضیح**: جابجایی کل نشان می‌دهد که یک ذره در طول کل شبیه‌سازی چقدر حرکت کرده است.
   - **تفسیر**: جابجایی کل بالاتر نشان می‌دهد که ذره در کل دوره شبیه‌سازی مسافت بیشتری را طی کرده است. برای مثال، جابجایی کل 7.687 برای ذره اول در مجموعه داده دوم به این معنی است که این ذره در کل دوره شبیه‌سازی مسافت بیشتری نسبت به جابجایی کل 3.191 در مجموعه داده اول طی کرده است.

### چگونه از این نتایج استفاده کنید:

1. **مقایسه پویایی ذرات بین سناریوها**:
   - شما می‌توانید مقادیر میانگین جابجایی و سرعت متوسط را برای ذرات مختلف در سناریوهای مختلف مقایسه کنید تا بفهمید کدام ذرات در کدام سناریوها فعال‌تر هستند.

2. **بررسی نوسانات حرکت**:
   - واریانس جابجایی می‌تواند به شما کمک کند تا تفاوت‌ها در نوسانات حرکت ذرات را در سناریوهای مختلف بررسی کنید. واریانس بالاتر به معنای حرکت ناپایدارتر و غیرقابل پیش‌بینی‌تر است.

3. **تحلیل جابجایی کلی**:
   - جابجایی کل می‌تواند به شما بگوید که کدام سناریوها باعث می‌شوند ذرات در کل دوره شبیه‌سازی بیشتر جابجا شوند، که ممکن است نشان‌دهنده تأثیر شرایط محیطی مختلف باشد.

### نتیجه‌گیری:

این معیارها به شما امکان می‌دهند تا به صورت کمی و قابل اندازه‌گیری تفاوت‌های بین سناریوهای مختلف را تحلیل کنید. با مقایسه این معیارها بین ذرات مختلف یا بین سناریوهای مختلف، می‌توانید به سوالاتی مانند "کدام سناریو باعث حرکت بیشتر ذرات می‌شود؟" یا "آیا حرکت ذرات در سناریوهای مختلف منظم‌تر است یا ناپایدارتر؟" پاسخ دهید.

اگر نیاز به تحلیل یا توضیحات بیشتری دارید، خوشحال می‌شوم که کمک کنم!
"""
# Main process

file_path = r"D:\PhD thesis\SimulationSV\cmake-build-debug\builds"
file1 = r"\Vesicles_movement.xyz"
file2 = r"\SynapsinI_movement.xyz"

file_path1 = file_path + time + file1
file_path2 = file_path + time + file2

# Example usage
input_files = [file_path1, file_path2]  # List of your input data files
base_directory = r'D:\PhD thesis\Data Analysis Code-Python'  # Set your base directory path

analysis_folder = create_analysis_folder(base_directory)
copy_input_files(input_files, analysis_folder)

print(f"All files have been organized and saved in: {analysis_folder}")
# Now you can proceed with your analyses and save all outputs to `analysis_folder`

data1 = read_xyz(file_path1)
data2 = read_xyz(file_path2)

# Calculate step sizes for both files
step_sizes1 = calculate_step_sizes(data1)
step_sizes2 = calculate_step_sizes(data2)

# Plot histograms for both files
plot_histogram(step_sizes2, 'Histogram of Step Sizes for SynapsinI Movement')
plot_histogram(step_sizes1, 'Histogram of Step Sizes for Vesicles Movement')

statistics1 = calculate_displacement_statistics(data1)
statistics2 = calculate_displacement_statistics(data2)
correlations = calculate_correlation(data1, data2)

print(f"Correlation for the first atom: {correlations[0]}")
print(f"Statistics for the first atom in dataset 1: {statistics1[0]}")
print(f"Statistics for the first atom in dataset 2: {statistics2[0]}")


correlation_matrix1 = calculate_correlation_matrix(data1)
plot_heatmap(correlation_matrix1)

correlation_matrix2 = calculate_correlation_matrix(data2)
plot_heatmap(correlation_matrix2)


grid_size = 23  # Adjust the grid size based on your spatial resolution needs
density_map1 = calculate_density(data1, grid_size)
density_map2 = calculate_density(data2, grid_size)

plot_density_map(density_map1, grid_size)
plot_density_map(density_map2, grid_size)

plot_trajectory(data1)  # Plot trajectory for a specific atom (e.g., atom 0)
plot_trajectory(data2)  # Plot trajectories for all atoms
correlation_matrix1 = calculate_correlation_matrix(data1)
plot_heatmap_and_save(correlation_matrix1, 'correlation_heatmap1.pdf')

correlation_matrix2 = calculate_correlation_matrix(data2)
plot_heatmap_and_save(correlation_matrix2, 'correlation_heatmap2.pdf')

msd1 = calculate_msd(data1)
msd2 = calculate_msd(data2)

plt.plot(msd1, label='MSD for Dataset 1')
plt.plot(msd2, label='MSD for Dataset 2')
plt.xlabel('Time Step')
plt.ylabel('Mean Square Displacement')
plt.legend()
plt.show()

labels1 = cluster_particles(data1, num_clusters=3)
print(labels1)

angles1 = calculate_angular_distribution(data1)
plt.hist(angles1, bins=30)
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.title('Angular Distribution of Displacements')
plt.show()

# After analyses are done, you can write a summary
summary_content = "This folder contains the outputs of the analysis performed on..."
write_summary_file(analysis_folder, summary_content)
