import numpy as np
def gaussian_noise(img, mean, sigma):
    '''
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
    '''
    # if img != '.DS_Store':
    # # 将图片灰度标准化
    #     img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
# 将噪声和图片叠加
    gaussian_out = img + noise
# 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, np.max(img))
# 将图片灰度范围的恢复为 0-255
#     gaussian_out = np.uint8(gaussian_out*255)
# 将噪声范围搞为 0-255
# noise = np.uint8(noise*255)
    return gaussian_out# 这里也会返回噪声，注意返回值
