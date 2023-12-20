import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import random


def getValue(filter_window, size):
    # 将窗口数组 filter_window 重新调整为大小为 size 的一维向量 target_vector
    target_vector = np.reshape(filter_window, (size,))
    # 对目标数组进行从小到大排序
    array = np.sort(target_vector)
    med = int(size / 2)
    top = int(size - 1)
    # 最小值
    minValue = array[0]
    # 计算中值
    medium = array[med]
    # 最大值
    maxValue = array[top]
    return int(minValue), int(medium), int(maxValue)  # 返回最小值、中值和最大值


def add_white_noise(image, noise_amplitude):
    # 生成与图像大小相同的随机噪声
    noise = np.random.normal(0, noise_amplitude, image.shape).astype(np.uint8)
    # 将噪声添加到图像中
    noisy_image = cv2.add(image, noise)

    return noisy_image


def add_salt_and_pepper_noise(image: np.ndarray, noise_ratio: float) -> np.ndarray:
    # 复制输入图像以保持原始数据不受修改
    noisy_image = np.copy(image)

    # 计算要添加的椒盐噪音的像素数量
    num_noise_pixels = int(noise_ratio * image.size)

    # 在随机位置添加椒盐噪音
    for _ in range(num_noise_pixels):
        # 随机抽取i，j位置
        i = random.randint(0, image.shape[0] - 1)
        # i，j的值的范围为0到image.shape[1] - 1
        j = random.randint(0, image.shape[1] - 1)
        for k in range(0, 3):
            # 随机添加白或黑像素
            noisy_image[i, j, k] = random.choice([0, 255])
    return noisy_image


def adaptive_median_filter2D(image_array, height, weight):
    window_sum = 0
    # 对输入二维数组的每个元素进行遍历
    for y in range(0, height):
        for x in range(0, weight):
            # 把窗口初始值设置为最小的大小值
            window_size = min_window
            # 当窗口达到最大值的时候跳出
            while window_size <= max_window:
                # size表示窗口的面积
                size = (2 * window_size + 1) * (2 * window_size + 1)
                # window_size参数指定了填充的大小，mode='constant'则表示采用常数模式进行填充。填充后的结果存储在名为array的新数组中。
                array = np.pad(image_array, window_size, mode='constant')
                # 从数组array中创建一个子窗口。该窗口是以(x, y)坐标为中心，大小为window_size * 2 + 1的正方形。
                filter_window = array[y:y + (window_size * 2) + 1, x:x + (window_size * 2) + 1]
                # 计算窗口的中值、最小值和最大值
                min, med, max = getValue(filter_window, size)
                # 如果min < med < max,也就是该点不是极值
                if min < med < max:
                    # 加int是有必要的，因为这能确保判断中不会报错overflow encountered in scalar subtract
                    # min < int(image_array[y, x]) < max说明该点不是噪音点
                    if min < int(image_array[y, x]) < max:
                        # 无需赋值，也就是取其原本的值
                        break
                    else:
                        image_array[y, x] = med
                        break
                # 如果该点为极值则大小加一继续判断是该点是不是噪音点
                else:
                    # print("Window increased")
                    window_size = window_size + 1
            window_sum = window_sum + window_size
    window_average = window_sum / (height * weight)
    return image_array, window_average


def adaptive_median_filter_show(noise_image, filename):
    global max_window, min_window
    image = cv2.imread(filename)
    # 因为接下来要展示噪音图像，使用复制一份副本进行操作
    # 以免噪音图像被意外修改
    copy_image = np.copy(noise_image)
    # shape[0]代表图像的高，shape[1]代表图像的宽
    height = noise_image.shape[0]
    weight = noise_image.shape[1]
    # 构造多个滤波图像与自适应中值滤波进行比较
    # 双边滤波
    bilateral_image = cv2.bilateralFilter(noise_image, 9, 75, 75)
    # 中值滤波
    median_image = cv2.medianBlur(noise_image, 5)
    # 高斯滤波
    gaussian_image = cv2.GaussianBlur(noise_image, (5, 5), 0)
    # 方框滤波
    box_image = cv2.boxFilter(noise_image, -1, (5, 5))
    # 自适应局部降低噪音滤波
    denoised_image = cv2.fastNlMeansDenoising(noise_image, h=10)
    # 用Matplotlib库的plt.subplots()函数创建了一个2行4列的子图布局，并将返回的Figure对象和Axes对象分别赋值给变量_和ax
    _, ax = plt.subplots(2, 4, figsize=(10, 10))
    # 如果是灰度图像
    if len(noise_image.shape) != 3:
        gray_image, ave_grey = adaptive_median_filter2D(noise_image, height, weight)
        print("灰度图像的滤波参数窗口大小为" + str(2 * ave_grey + 1) + "*" + str(2 * ave_grey + 1))
        # 自适应中值滤波图像
        # 将灰度图像转换为 RGB 灰度图像。
        # 也就是灰度模式读取图像，这将将图像的深度设置为 CV_64F
        rgb_gray_image = cv2.convertScaleAbs(gray_image)
        # 将 RGB 灰度图像转换为 BGR 彩色图像。
        rgb_gray_image = cv2.cvtColor(rgb_gray_image, cv2.COLOR_GRAY2BGR)
        ax[1][0].imshow(rgb_gray_image)
        ax[1][0].title.set_text("adaptive_median_filter  window_size" + str(2 * ave_grey + 1) + "*" + str(2 * ave_grey + 1))
    # 如果是彩色图像
    else:
        # 分离rbg通道
        # 注意！！！通道的排序为blue/green/red，而不是按rbg的顺序排序
        B = copy_image[:, :, 0]
        G = copy_image[:, :, 1]
        R = copy_image[:, :, 2]
        # 对R通道进行自适应中值滤波
        print("R")
        R1, ave_r = adaptive_median_filter2D(R, height, weight)
        # 对G通道进行自适应中值滤波
        print("G")
        G1, ave_g = adaptive_median_filter2D(G, height, weight)
        # 对B通道进行自适应中值滤波
        print("B")
        B1, ave_b = adaptive_median_filter2D(B, height, weight)
        ave_window = round(2 * (ave_r+ave_g+ave_b)/3 + 1)
        print("彩色图像的滤波参数窗口大小为:" + str(ave_window) + "*" + str(ave_window))
        # 开始进行组合，将经过自适应中值滤波的通道图像（OB、OG和OR）赋值给 copy_image 的对应多通道图像
        # 切记，不要搞反了顺序，是BGR。
        copy_image[:, :, 0] = B1
        copy_image[:, :, 1] = G1
        copy_image[:, :, 2] = R1
        # 不这样的话输出图像会变成灰度图像或者单一颜色图像
        # 自适应中值滤波图像
        # 将 BGR 图像转换为 RGB 图像。
        rgb_copy_image = cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB)
        ax[1][0].imshow(rgb_copy_image)
        ax[1][0].title.set_text("adaptive_median_filter  window_size" + str(ave_window) + "*" + str(ave_window))
    # 初始图像
    # cv2.cvtColor()函数将图像median_image从BGR颜色空间转换为RGB
    # 将 BGR 图像转换为 RGB 图像。
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax[0][0].imshow(rgb_image)
    ax[0][0].title.set_text("origin")
    # 噪音图像
    # 将 BGR 图像转换为 RGB 图像。
    rgb_noise_image = cv2.cvtColor(noise_image, cv2.COLOR_BGR2RGB)
    ax[0][1].imshow(rgb_noise_image)
    ax[0][1].title.set_text("noise_image")
    # 高斯滤波图像
    # 将 BGR 图像转换为 RGB 图像。
    rgb_gaussian_image = cv2.cvtColor(gaussian_image, cv2.COLOR_BGR2RGB)
    ax[0][2].imshow(rgb_gaussian_image)
    ax[0][2].title.set_text("gaussian_image")
    # 自适应局部降噪滤波滤波图像
    # 将 BGR 图像转换为 RGB 图像。
    rgb_denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
    ax[0][3].imshow(rgb_denoised_image)
    ax[0][3].title.set_text("denoised_image")
    # 双边滤波
    # 将 BGR 图像转换为 RGB 图像。
    rgb_bilateral_image = cv2.cvtColor(bilateral_image, cv2.COLOR_BGR2RGB)
    ax[1][1].imshow(rgb_bilateral_image)
    ax[1][1].title.set_text("bilateral_image")
    # 中值滤波
    # 将 BGR 图像转换为 RGB 图像。
    rgb_medium_image = cv2.cvtColor(median_image, cv2.COLOR_BGR2RGB)
    ax[1][2].imshow(rgb_medium_image)
    ax[1][2].title.set_text("medium_image")
    # 方框滤波
    # 将 BGR 图像转换为 RGB 图像。
    rgb_box_image = cv2.cvtColor(box_image, cv2.COLOR_BGR2RGB)
    ax[1][3].imshow(rgb_box_image)
    ax[1][3].title.set_text("box_image")
    plt.show()

    return copy_image


# 读取图像路径
filename = 'C:/Users/18133/Desktop/wiener/10.jpg'
# 定义最大窗口
min_window = 1
# 定义最小窗口
max_window = 4
# 从指定的文件名加载图像。
image = cv2.imread(filename)
# 得到椒盐噪音图像
salt_noise_image = add_salt_and_pepper_noise(image, 0.2)
# 得到白噪音图像
white_noise_image = add_white_noise(image, 2)
# 以椒盐噪音为噪音图像底模进行滤波
img = adaptive_median_filter_show(salt_noise_image, filename)
# 以白噪音为噪音图像底模进行滤波
img2 = adaptive_median_filter_show(white_noise_image, filename)
