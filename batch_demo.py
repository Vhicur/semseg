import os
import subprocess

# 设置图片文件夹路径
image_folder = r'D:\NanJing\jiejing_point\nanjing_point'

# 设置配置文件路径
config_file = 'config/cityscapes/cityscapes_pspnet50.yaml'

# 获取图片文件夹中的所有图片文件
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

# 设置环境变量 PYTHONPATH
current_env = os.environ.copy()
current_env['PYTHONPATH'] = './'

# 遍历每张图片并运行 demo.py
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    command = f'PYTHONPATH=./ python tool/demo.py --config {config_file} --image "{image_path}" --weight initmodel/pspnet50.pth'
    subprocess.run(command, shell=True, env=current_env)