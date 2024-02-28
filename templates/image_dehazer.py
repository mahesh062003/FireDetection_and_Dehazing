import cv2
import numpy as np

def remove_haze(image, regularize_lambda=0.05, sigma=1, delta=1, showHazeTransmissionMap=False):
    I = image / 255.0

    dark_channel = get_dark_channel(I)
    atmospheric_light = get_atmospheric_light(I, dark_channel)

    transmission = get_transmission(I, atmospheric_light, regularize_lambda, sigma, delta)
    transmission = np.clip(transmission, 0.1, 1.0)

    J = recover_scene_radiance(I, transmission, atmospheric_light)

    if showHazeTransmissionMap:
        return J, transmission
    else:
        return J

def get_dark_channel(image, patch_size=15):
    return np.min(cv2.boxFilter(image, -1, (patch_size, patch_size)), axis=2)

def get_atmospheric_light(image, dark_channel):
    flat_dark_channel = dark_channel.flatten()
    num_pixels = flat_dark_channel.size
    num_brightest = int(num_pixels * 0.001)  # Select top 0.1% brightest pixels

    indices = np.argpartition(flat_dark_channel, -num_brightest)[-num_brightest:]
    bright_pixels = image.reshape(-1, 3)[indices]

    atmospheric_light = np.max(bright_pixels, axis=0)
    return atmospheric_light

def get_transmission(image, atmospheric_light, regularize_lambda, sigma, delta):
    normalized_image = image / atmospheric_light

    intensity = 0.299 * normalized_image[:, :, 0] + 0.587 * normalized_image[:, :, 1] + 0.114 * normalized_image[:, :, 2]

    mean_intensity = cv2.boxFilter(intensity, -1, (15, 15))
    mean_intensity = np.maximum(mean_intensity, delta)

    transmission = 1 - regularize_lambda * (intensity / mean_intensity)
    transmission = np.clip(transmission, 0, 1)

    return cv2.ximgproc.guidedFilter(image, transmission, 40, 1e-3)

def recover_scene_radiance(image, transmission, atmospheric_light):
    J = np.zeros_like(image)
    for i in range(3):
        J[:, :, i] = (image[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
    return np.clip(J, 0, 1)
