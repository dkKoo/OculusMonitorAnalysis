import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilenames, askdirectory
from pykalman import KalmanFilter
import seaborn as sns
import pandas as pd
import time

def kalman_filter(data):
    # 초기 상태 설정
    initial_state_mean = data[0]
    n_dim = len(initial_state_mean)
    transition_matrix = np.eye(n_dim)
    observation_matrix = np.eye(n_dim)
    observation_covariance = np.eye(n_dim) * 1e-4
    transition_covariance = np.eye(n_dim) * 1e-6

    # 칼만 필터 초기화
    kf = KalmanFilter(transition_matrices=transition_matrix,
                      observation_matrices=observation_matrix,
                      initial_state_mean=initial_state_mean,
                      observation_covariance=observation_covariance,
                      transition_covariance=transition_covariance)

    # 칼만 필터 적용
    filtered_state_means, _ = kf.filter(data)

    return filtered_state_means

def process_data(file_paths):
    # 데이터를 저장할 리스트 초기화
    time_data_list = []
    pos_data_list = []
    orientation_data_list = []
    velocity_data_list = []

    for file_path in file_paths:
        time_data = []
        pos_data = []
        orientation_data = []
        velocity_data = []

        # CSV 파일 읽기
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            
            # 헤더 행 건너뛰기
            next(csv_reader)
            
            # 데이터 읽어오기
            for row in csv_reader:
                time_data.append(float(row[0]))
                pos_data.append([float(row[22]), float(row[23]), float(row[24])])
                orientation_data.append([float(row[25]), float(row[26]), float(row[27]), float(row[28])])

        # 위치 데이터를 numpy 배열로 변환
        pos_data = np.array(pos_data)

        # 방향 데이터를 numpy 배열로 변환
        orientation_data = np.array(orientation_data)

        # 칼만 필터를 사용하여 위치 데이터 필터링
        filtered_pos_data = kalman_filter(pos_data)

        # 칼만 필터를 사용하여 방향 데이터 필터링
        filtered_orientation_data = kalman_filter(orientation_data)

        # 속도 계산
        velocity_data = np.diff(filtered_pos_data, axis=0) / np.diff(time_data)[:, np.newaxis]

        time_data_list.append(time_data)
        pos_data_list.append(filtered_pos_data)
        orientation_data_list.append(filtered_orientation_data)
        velocity_data_list.append(velocity_data)

    return time_data_list, pos_data_list, orientation_data_list, velocity_data_list

def save_data(time_data_list, pos_data_list, orientation_data_list, velocity_data_list, output_dir):
    for i in range(len(time_data_list)):
        file_name = f'processed_data_{i+1}.csv'
        file_path = os.path.join(output_dir, file_name)
        
        with open(file_path, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['Time', 'HeadPosX', 'HeadPosY', 'HeadPosZ', 'HeadOrientationW', 'HeadOrientationX', 'HeadOrientationY', 'HeadOrientationZ', 'VelocityX', 'VelocityY', 'VelocityZ'])
            
            for j in range(len(time_data_list[i])):
                row = [time_data_list[i][j]]
                row.extend(pos_data_list[i][j])
                row.extend(orientation_data_list[i][j])
                if j < len(velocity_data_list[i]):
                    row.extend(velocity_data_list[i][j])
                else:
                    row.extend([0, 0, 0])  # 마지막 행의 속도 데이터는 없으므로 0으로 채움
                csv_writer.writerow(row)

def calculate_stats(time_data_list, pos_data_list, orientation_data_list, velocity_data_list):
    stats_list = []

    for i in range(len(time_data_list)):
        stats = {}

        # 위치 데이터 통계
        stats['HeadPosX'] = {'max': np.max(pos_data_list[i][:, 0]), 'min': np.min(pos_data_list[i][:, 0]), 'mean': np.mean(pos_data_list[i][:, 0]),
                             'std': np.std(pos_data_list[i][:, 0]), 'cv': np.std(pos_data_list[i][:, 0]) / np.mean(pos_data_list[i][:, 0]),
                             'integral': np.trapz(pos_data_list[i][:, 0], time_data_list[i])}
        stats['HeadPosY'] = {'max': np.max(pos_data_list[i][:, 1]), 'min': np.min(pos_data_list[i][:, 1]), 'mean': np.mean(pos_data_list[i][:, 1]),
                             'std': np.std(pos_data_list[i][:, 1]), 'cv': np.std(pos_data_list[i][:, 1]) / np.mean(pos_data_list[i][:, 1]),
                             'integral': np.trapz(pos_data_list[i][:, 1], time_data_list[i])}
        stats['HeadPosZ'] = {'max': np.max(pos_data_list[i][:, 2]), 'min': np.min(pos_data_list[i][:, 2]), 'mean': np.mean(pos_data_list[i][:, 2]),
                             'std': np.std(pos_data_list[i][:, 2]), 'cv': np.std(pos_data_list[i][:, 2]) / np.mean(pos_data_list[i][:, 2]),
                             'integral': np.trapz(pos_data_list[i][:, 2], time_data_list[i])}

        # 방향 데이터 통계
        stats['HeadOrientationW'] = {'max': np.max(orientation_data_list[i][:, 0]), 'min': np.min(orientation_data_list[i][:, 0]), 'mean': np.mean(orientation_data_list[i][:, 0]),
                                     'std': np.std(orientation_data_list[i][:, 0]), 'cv': np.std(orientation_data_list[i][:, 0]) / np.mean(orientation_data_list[i][:, 0])}
        stats['HeadOrientationX'] = {'max': np.max(orientation_data_list[i][:, 1]), 'min': np.min(orientation_data_list[i][:, 1]), 'mean': np.mean(orientation_data_list[i][:, 1]),
                                     'std': np.std(orientation_data_list[i][:, 1]), 'cv': np.std(orientation_data_list[i][:, 1]) / np.mean(orientation_data_list[i][:, 1])}
        stats['HeadOrientationY'] = {'max': np.max(orientation_data_list[i][:, 2]), 'min': np.min(orientation_data_list[i][:, 2]), 'mean': np.mean(orientation_data_list[i][:, 2]),
                                     'std': np.std(orientation_data_list[i][:, 2]), 'cv': np.std(orientation_data_list[i][:, 2]) / np.mean(orientation_data_list[i][:, 2])}
        stats['HeadOrientationZ'] = {'max': np.max(orientation_data_list[i][:, 3]), 'min': np.min(orientation_data_list[i][:, 3]), 'mean': np.mean(orientation_data_list[i][:, 3]),
                                     'std': np.std(orientation_data_list[i][:, 3]), 'cv': np.std(orientation_data_list[i][:, 3]) / np.mean(orientation_data_list[i][:, 3])}

        # 속도 데이터 통계
        stats['VelocityX'] = {'max': np.max(velocity_data_list[i][:, 0]), 'min': np.min(velocity_data_list[i][:, 0]), 'mean': np.mean(velocity_data_list[i][:, 0]),
                              'std': np.std(velocity_data_list[i][:, 0]), 'cv': np.std(velocity_data_list[i][:, 0]) / np.mean(velocity_data_list[i][:, 0]),
                              'integral': np.trapz(velocity_data_list[i][:, 0], time_data_list[i][:-1])}
        stats['VelocityY'] = {'max': np.max(velocity_data_list[i][:, 1]), 'min': np.min(velocity_data_list[i][:, 1]), 'mean': np.mean(velocity_data_list[i][:, 1]),
                              'std': np.std(velocity_data_list[i][:, 1]), 'cv': np.std(velocity_data_list[i][:, 1]) / np.mean(velocity_data_list[i][:, 1]),
                              'integral': np.trapz(velocity_data_list[i][:, 1], time_data_list[i][:-1])}
        stats['VelocityZ'] = {'max': np.max(velocity_data_list[i][:, 2]), 'min': np.min(velocity_data_list[i][:, 2]), 'mean': np.mean(velocity_data_list[i][:, 2]),
                              'std': np.std(velocity_data_list[i][:, 2]), 'cv': np.std(velocity_data_list[i][:, 2]) / np.mean(velocity_data_list[i][:, 2]),
                              'integral': np.trapz(velocity_data_list[i][:, 2], time_data_list[i][:-1])}

        # 속도 크기 (speed) 계산
        speed_data = np.linalg.norm(velocity_data_list[i], axis=1)

        # 속도 크기 (speed) 데이터 통계
        stats['Speed'] = {'max': np.max(speed_data), 'min': np.min(speed_data), 'mean': np.mean(speed_data),
                          'std': np.std(speed_data), 'cv': np.std(speed_data) / np.mean(speed_data),
                          'integral': np.trapz(speed_data, time_data_list[i][:-1])}

        # 상관관계 계산
        stats['Correlation'] = {}
        stats['Correlation']['HeadPosXY'] = np.corrcoef(pos_data_list[i][:, 0], pos_data_list[i][:, 1])[0, 1]
        stats['Correlation']['HeadPosXZ'] = np.corrcoef(pos_data_list[i][:, 0], pos_data_list[i][:, 2])[0, 1]
        stats['Correlation']['HeadPosYZ'] = np.corrcoef(pos_data_list[i][:, 1], pos_data_list[i][:, 2])[0, 1]
        stats['Correlation']['VelocityXY'] = np.corrcoef(velocity_data_list[i][:, 0], velocity_data_list[i][:, 1])[0, 1]
        stats['Correlation']['VelocityXZ'] = np.corrcoef(velocity_data_list[i][:, 0], velocity_data_list[i][:, 2])[0, 1]
        stats['Correlation']['VelocityYZ'] = np.corrcoef(velocity_data_list[i][:, 1], velocity_data_list[i][:, 2])[0, 1]

        for key, value in stats.items():
            if key != 'Correlation':
                for subkey, subvalue in value.items():
                    stats[key][subkey] = round(subvalue, 3)
            else:
                for subkey, subvalue in value.items():
                    stats[key][subkey] = round(subvalue, 3)

        stats_list.append(stats)

    return stats_list

def save_stats(stats_list, output_dir):
    # 각 변수들의 평균과 표준편차 계산
    stats_mean_std = {}
    for key in stats_list[0].keys():
        if key != 'Correlation':
            stats_mean_std[key] = {}
            for subkey in stats_list[0][key].keys():
                values = [stats[key][subkey] for stats in stats_list]
                stats_mean_std[key][subkey] = {'mean': np.mean(values), 'std': np.std(values)}

    file_name = 'stats.csv'
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Variable', 'Max_Mean', 'Max_Std', 'Min_Mean', 'Min_Std', 'Mean_Mean', 'Mean_Std',
                             'Std_Mean', 'Std_Std', 'CV_Mean', 'CV_Std', 'Integral_Mean', 'Integral_Std'])
        for key, value in stats_mean_std.items():
            row = [key]
            for subkey in ['max', 'min', 'mean', 'std', 'cv', 'integral']:
                if subkey in value:
                    row.extend([value[subkey]['mean'], value[subkey]['std']])
                else:
                    row.extend(['', ''])
            csv_writer.writerow(row)
        
        csv_writer.writerow([])
        csv_writer.writerow(['Correlation', 'Value'])
        for corr_key, corr_value in stats_list[0]['Correlation'].items():
            values = [stats['Correlation'][corr_key] for stats in stats_list]
            csv_writer.writerow([corr_key, np.mean(values)])

def calculate_mean_std(data_list):
    min_length = min(len(data) for data in data_list)
    data_array = np.array([data[:min_length] for data in data_list])
    mean_data = np.mean(data_array, axis=0)
    std_data = np.std(data_array, axis=0)
    return mean_data, std_data

def plot_data(time_data_list, pos_data_list, orientation_data_list, velocity_data_list, output_dir):
    # 평균 데이터 계산
    mean_pos_data, std_pos_data = calculate_mean_std(pos_data_list)
    mean_orientation_data, std_orientation_data = calculate_mean_std(orientation_data_list)
    mean_velocity_data, std_velocity_data = calculate_mean_std(velocity_data_list)

    # time_data_list에서 가장 작은 크기 찾기
    min_length = min(len(data) for data in time_data_list)
    time_data = time_data_list[0][:min_length]

    # 쿼터니언 값의 시계열 그래프
    plt.figure()
    plt.plot(time_data, mean_orientation_data[:, 0], label='W')
    plt.plot(time_data, mean_orientation_data[:, 1], label='X')
    plt.plot(time_data, mean_orientation_data[:, 2], label='Y')
    plt.plot(time_data, mean_orientation_data[:, 3], label='Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Quaternion Value')
    plt.legend()
    plt.tight_layout()
    plot_file_name = 'quaternion_timeseries_mean.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

    # 속도 데이터 플롯
    plt.figure()
    plt.plot(time_data[:-1], mean_velocity_data[:, 0], label='VelocityX')
    plt.plot(time_data[:-1], mean_velocity_data[:, 1], label='VelocityY')
    plt.plot(time_data[:-1], mean_velocity_data[:, 2], label='VelocityZ')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.tight_layout()
    plot_file_name = 'velocity_mean.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

    # 위치 데이터의 3D 트레이스 플롯
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(mean_pos_data[:, 0], mean_pos_data[:, 1], mean_pos_data[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Position Trace')
    plt.tight_layout()
    plot_file_name = 'position_trace_3d_mean.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

    # 속도 크기 플롯
    velocity_magnitude = np.linalg.norm(mean_velocity_data, axis=1)
    plt.figure()
    plt.plot(time_data[:-1], velocity_magnitude)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Magnitude (m/s)')
    plt.title('Velocity Magnitude')
    plt.tight_layout()
    plot_file_name = 'velocity_magnitude_mean.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

    # 가속도 플롯
    acceleration_data = np.diff(mean_velocity_data, axis=0) / np.diff(time_data[:-1])[:, np.newaxis]
    plt.figure()
    plt.plot(time_data[:-2], acceleration_data[:, 0], label='AccelerationX')
    plt.plot(time_data[:-2], acceleration_data[:, 1], label='AccelerationY')
    plt.plot(time_data[:-2], acceleration_data[:, 2], label='AccelerationZ')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()
    plt.tight_layout()
    plot_file_name = 'acceleration_mean.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

    # 히스토그램
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].hist(mean_pos_data[:, 0], bins=20)
    axs[0, 0].set_title('Position X')
    axs[0, 1].hist(mean_pos_data[:, 1], bins=20)
    axs[0, 1].set_title('Position Y')
    axs[1, 0].hist(mean_pos_data[:, 2], bins=20)
    axs[1, 0].set_title('Position Z')
    axs[1, 1].hist(velocity_magnitude, bins=20)
    axs[1, 1].set_title('Velocity Magnitude')
    plt.tight_layout()
    plot_file_name = 'histograms_mean.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

    # 산점도 매트릭스
    scatter_matrix = pd.DataFrame(np.hstack((mean_pos_data[:-1], mean_velocity_data)), columns=['PosX', 'PosY', 'PosZ', 'VelX', 'VelY', 'VelZ'])
    sns.set(style="ticks")
    sns.pairplot(scatter_matrix, diag_kind="kde")
    plot_file_name = 'scatter_matrix_seaborn_mean.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

# tkinter 창 생성
root = Tk()
root.withdraw()

# 파일 경로 설정
file_paths = askopenfilenames(title='Select Input Files')

# 출력 디렉토리 설정
output_dir = askdirectory(title='Select Output Directory')

if file_paths and output_dir:
    start_time = time.time()  # 코드 실행 시작 시간 측정
    
    time_data_list, pos_data_list, orientation_data_list, velocity_data_list = process_data(file_paths)
    save_data(time_data_list, pos_data_list, orientation_data_list, velocity_data_list, output_dir)
    stats_list = calculate_stats(time_data_list, pos_data_list, orientation_data_list, velocity_data_list)
    save_stats(stats_list, output_dir)
    plot_data(time_data_list, pos_data_list, orientation_data_list, velocity_data_list, output_dir)
    
    end_time = time.time()  # 코드 실행 종료 시간 측정
    execution_time = end_time - start_time  # 코드 실행 시간 계산
    
    print(f"데이터 처리 및 분석이 완료되었습니다. 출력 파일과 그래프는 선택한 출력 디렉토리에서 확인할 수 있습니다.")
    print(f"코드 실행 시간: {execution_time:.3f}초")