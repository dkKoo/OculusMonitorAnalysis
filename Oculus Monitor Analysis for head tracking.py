import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

def process_data(file_path):
    # 데이터를 저장할 리스트 초기화
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

    # 속도 계산
    velocity_data = np.diff(pos_data, axis=0) / np.diff(time_data)[:, np.newaxis]

    return time_data, pos_data, orientation_data, velocity_data

def save_data(time_data, pos_data, orientation_data, velocity_data, output_dir):
    file_name = 'processed_data.csv'
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Time', 'HeadPosX', 'HeadPosY', 'HeadPosZ', 'HeadOrientationW', 'HeadOrientationX', 'HeadOrientationY', 'HeadOrientationZ', 'VelocityX', 'VelocityY', 'VelocityZ'])
        
        for i in range(len(time_data)):
            row = [time_data[i]]
            row.extend(pos_data[i])
            row.extend(orientation_data[i])
            if i < len(velocity_data):
                row.extend(velocity_data[i])
            else:
                row.extend([0, 0, 0])  # 마지막 행의 속도 데이터는 없으므로 0으로 채움
            csv_writer.writerow(row)

def calculate_stats(time_data, pos_data, orientation_data, velocity_data):
    stats = {}

    # 위치 데이터 통계
    stats['HeadPosX'] = {'max': np.max(pos_data[:, 0]), 'min': np.min(pos_data[:, 0]), 'mean': np.mean(pos_data[:, 0])}
    stats['HeadPosY'] = {'max': np.max(pos_data[:, 1]), 'min': np.min(pos_data[:, 1]), 'mean': np.mean(pos_data[:, 1])}
    stats['HeadPosZ'] = {'max': np.max(pos_data[:, 2]), 'min': np.min(pos_data[:, 2]), 'mean': np.mean(pos_data[:, 2])}

    # 방향 데이터 통계
    stats['HeadOrientationW'] = {'max': np.max(orientation_data[:, 0]), 'min': np.min(orientation_data[:, 0]), 'mean': np.mean(orientation_data[:, 0])}
    stats['HeadOrientationX'] = {'max': np.max(orientation_data[:, 1]), 'min': np.min(orientation_data[:, 1]), 'mean': np.mean(orientation_data[:, 1])}
    stats['HeadOrientationY'] = {'max': np.max(orientation_data[:, 2]), 'min': np.min(orientation_data[:, 2]), 'mean': np.mean(orientation_data[:, 2])}
    stats['HeadOrientationZ'] = {'max': np.max(orientation_data[:, 3]), 'min': np.min(orientation_data[:, 3]), 'mean': np.mean(orientation_data[:, 3])}

    # 속도 데이터 통계
    stats['VelocityX'] = {'max': np.max(velocity_data[:, 0]), 'min': np.min(velocity_data[:, 0]), 'mean': np.mean(velocity_data[:, 0])}
    stats['VelocityY'] = {'max': np.max(velocity_data[:, 1]), 'min': np.min(velocity_data[:, 1]), 'mean': np.mean(velocity_data[:, 1])}
    stats['VelocityZ'] = {'max': np.max(velocity_data[:, 2]), 'min': np.min(velocity_data[:, 2]), 'mean': np.mean(velocity_data[:, 2])}

    # 위치 데이터 CV 계산
    stats['HeadPosX']['cv'] = np.std(pos_data[:, 0]) / np.mean(pos_data[:, 0])
    stats['HeadPosY']['cv'] = np.std(pos_data[:, 1]) / np.mean(pos_data[:, 1])
    stats['HeadPosZ']['cv'] = np.std(pos_data[:, 2]) / np.mean(pos_data[:, 2])

    # 방향 데이터 CV 계산
    stats['HeadOrientationW']['cv'] = np.std(orientation_data[:, 0]) / np.mean(orientation_data[:, 0])
    stats['HeadOrientationX']['cv'] = np.std(orientation_data[:, 1]) / np.mean(orientation_data[:, 1])
    stats['HeadOrientationY']['cv'] = np.std(orientation_data[:, 2]) / np.mean(orientation_data[:, 2])
    stats['HeadOrientationZ']['cv'] = np.std(orientation_data[:, 3]) / np.mean(orientation_data[:, 3])

    # 속도 데이터 CV 계산
    stats['VelocityX']['cv'] = np.std(velocity_data[:, 0]) / np.mean(velocity_data[:, 0])
    stats['VelocityY']['cv'] = np.std(velocity_data[:, 1]) / np.mean(velocity_data[:, 1])
    stats['VelocityZ']['cv'] = np.std(velocity_data[:, 2]) / np.mean(velocity_data[:, 2])

    return stats

def save_stats(stats, output_dir):
    file_name = 'stats.csv'
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Variable', 'Max', 'Min', 'Mean', 'CV'])
        for key, value in stats.items():
            csv_writer.writerow([key, value['max'], value['min'], value['mean'], value['cv']])

def plot_data(time_data, pos_data, orientation_data, velocity_data, output_dir):
    # 위치 데이터 플롯 (3차원)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_data[:, 0], pos_data[:, 1], pos_data[:, 2], label='Head Position')
    ax.set_xlabel('HeadPosX (m)')
    ax.set_ylabel('HeadPosY (m)')
    ax.set_zlabel('HeadPosZ (m)')
    ax.legend()
    plt.tight_layout()
    plot_file_name = 'head_position_3d.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

    # HeadPosX 플롯 (2차원)
    plt.figure()
    plt.plot(time_data, pos_data[:, 0], label='HeadPosX')
    plt.xlabel('Time (s)')
    plt.ylabel('HeadPosX (m)')
    plt.legend()
    plt.tight_layout()
    plot_file_name = 'head_position_x.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

    # HeadPosY 플롯 (2차원)
    plt.figure()
    plt.plot(time_data, pos_data[:, 1], label='HeadPosY')
    plt.xlabel('Time (s)')
    plt.ylabel('HeadPosY (m)')
    plt.legend()
    plt.tight_layout()
    plot_file_name = 'head_position_y.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

    # HeadPosZ 플롯 (2차원)
    plt.figure()
    plt.plot(time_data, pos_data[:, 2], label='HeadPosZ')
    plt.xlabel('Time (s)')
    plt.ylabel('HeadPosZ (m)')
    plt.legend()
    plt.tight_layout()
    plot_file_name = 'head_position_z.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

    # 방향 데이터 플롯 (3차원)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(orientation_data[:, 1], orientation_data[:, 2], orientation_data[:, 3], label='Head Orientation')
    ax.set_xlabel('HeadOrientationX (quaternion)')
    ax.set_ylabel('HeadOrientationY (quaternion)')
    ax.set_zlabel('HeadOrientationZ (quaternion)')
    ax.legend()
    plt.tight_layout()
    plot_file_name = 'head_orientation_3d.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

    # 방향 데이터 플롯 (2차원)
    plt.figure()
    plt.plot(time_data, orientation_data[:, 1], label='HeadOrientationX')
    plt.plot(time_data, orientation_data[:, 2], label='HeadOrientationY')
    plt.plot(time_data, orientation_data[:, 3], label='HeadOrientationZ')
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation (quaternion)')
    plt.legend()
    plt.tight_layout()
    plot_file_name = 'head_orientation_xyz.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

    plt.figure()
    plt.plot(time_data, orientation_data[:, 0], label='HeadOrientationW')
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation (quaternion)')
    plt.legend()
    plt.tight_layout()
    plot_file_name = 'head_orientation_w.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

    # 속도 데이터 플롯
    plt.figure()
    plt.plot(time_data[:-1], velocity_data[:, 0], label='VelocityX')
    plt.plot(time_data[:-1], velocity_data[:, 1], label='VelocityY')
    plt.plot(time_data[:-1], velocity_data[:, 2], label='VelocityZ')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.tight_layout()
    plot_file_name = 'velocity.png'
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

# tkinter 창 생성
root = Tk()
root.withdraw()

# 파일 경로 설정
file_path = askopenfilename(title='Select Input File')

# 출력 디렉토리 설정
output_dir = askdirectory(title='Select Output Directory')

if file_path and output_dir:
    time_data, pos_data, orientation_data, velocity_data = process_data(file_path)
    save_data(time_data, pos_data, orientation_data, velocity_data, output_dir)
    stats = calculate_stats(time_data, pos_data, orientation_data, velocity_data)
    print("통계:")
    for key, value in stats.items():
        print(f"{key}: 최대값={value['max']}, 최소값={value['min']}, 평균값={value['mean']}, CV={value['cv']}")
    save_stats(stats, output_dir)
    plot_data(time_data, pos_data, orientation_data, velocity_data, output_dir)
else:
    print("파일 경로 또는 출력 디렉토리가 선택되지 않았습니다.")
