import numpy as np
import PIL
import matplotlib.pyplot as plt
import os

def change_brightness(img, brightness):
    brightness = np.clip(int(brightness), -255, 255)
    new_image = np.clip(img.astype(np.uint16) + brightness, 0, 255)
    return new_image.astype(np.uint8)

def change_contrast(img, contrast):
    contrast = np.clip(float(contrast), -255, 255)
    factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
    new_image = np.clip(factor * (img.astype(float) - 128) + 128, 0, 255)
    return new_image.astype(np.uint8)

def flip_image(img, dir=1):
    if dir == 1:
        return np.flip(img, axis=0).astype(np.uint8)
    else:
        return np.flip(img, axis=1).astype(np.uint8)
    
def grayscale_image(img):
    return np.dot(img[..., :3], [0.3, 0.59, 0.11]).astype(np.uint8)

def sepia_image(img):
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])
    return np.clip(np.dot(img, sepia_matrix.T), 0, 255).astype(np.uint8)

def kernel_1d(size, sigma):
    x = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def convolution_channel(channel, kernel):   
    result = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=0, arr=channel)
    result = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=result)
    return result

def blur_image(img, size, sigma):
    kernel = kernel_1d(size, sigma)
    result = np.ones_like(img, dtype=np.float64) * 255
    
    # For grayscale images
    if(img.shape[-1] != 3):
        result[:, :] = convolution_channel(img[:,:], kernel)
    else:
        for c in range(img.shape[-1]):
            result[:, :, c] = convolution_channel(img[:,:,c], kernel)

    return np.uint8(result)

def sharp_image(img, size, sigma):
    identity = np.zeros(shape=size)
    identity[size//2] = 1
    kernel = 2*identity - kernel_1d(size, sigma)
    result = np.ones_like(img, dtype=np.float64) * 255
    
    if(img.shape[-1] != 3):
        result[:, :] = convolution_channel(img[:,:], kernel)
    else:
        for c in range(img.shape[-1]):
            result[:, :, c] = convolution_channel(img[:,:,c], kernel)
        
    result = np.clip(result, 0, 255)
    return np.uint8(result)

def zoom_image(img, percentage):
    new_height = int(img.shape[0] * percentage)
    new_width = int(img.shape[1] * percentage)
    
    center_row = img.shape[0] // 2
    center_col = img.shape[1] // 2
    
    col_start = center_col - new_width // 2
    col_end = col_start + new_width
    
    row_start = center_row - new_height // 2
    row_end = row_start + new_height
    
    if(img.shape[-1] == 3):
        return img[row_start:row_end, col_start:col_end, :]
    else:
        return img[row_start:row_end, col_start:col_end]

def crop_circle(img):
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    
    radius = min(center_x, center_y)

    mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
    cropped_image = img.copy()
    cropped_image[~mask] = 0
    return cropped_image

def crop_butterfly(img, elipse_size=8):
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2

    elipse = (img.shape[1] / elipse_size, 
              img.shape[0] / elipse_size, 
              img.shape[1] - img.shape[1] / elipse_size, 
              img.shape[0] - img.shape[0] / elipse_size)
    
    a = np.linalg.norm(np.array([elipse[0], elipse[1]]) - np.array([elipse[2], -elipse[3]])) / 2
    b = np.sqrt(a**2 - (np.linalg.norm(np.array([elipse[0], elipse[1]]) - np.array([elipse[2], elipse[3]])) / 2)**2)
    
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    ellipse_rotation = 45
    x_rot = (x - center_x) * np.cos(np.radians(ellipse_rotation)) - (y - center_y) * np.sin(np.radians(ellipse_rotation))
    y_rot = (x - center_x) * np.sin(np.radians(ellipse_rotation)) + (y - center_y) * np.cos(np.radians(ellipse_rotation))
    
    mask1 = (x_rot)**2 / a**2 + (y_rot) **2 / b**2 <= 1
    mask2 = np.flip(mask1, axis=0)
    mask = np.logical_or(mask1, mask2)
    cropped_img = img.copy()
    cropped_img[~mask] = 0
    return cropped_img

def save_img(img, filepath, suffix):
    output_img = PIL.Image.fromarray(img)
    
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    os.makedirs('./output/', exist_ok=True)
    output_img.save('./output/' + file_name + suffix + '.png')

def main():
    file_path = input("Enter file path: ")
    image = PIL.Image.open(file_path)
    img = np.array(image, dtype=np.uint8)
    original_img = img.copy()
    
    choose = 0
    
    while (choose != -1):
        print("1: Change brightness")
        print("2: Change contrast")
        print("3: Flip image")
        print("4: Turn into grayscale image")
        print("5: Turn into sepia image")
        print("6: Blur image")
        print("7: Sharpen image")
        print("8: Zoom image")
        print("9: Crop image into a circle")
        choose = int(input("Enter your selection (-1 to quit, 0 to run all): "))
        
        if(choose == 1):
            brightness = int(input("Enter the brightness: "))
            new_img = change_brightness(original_img, brightness=brightness)
            save_img(new_img, file_path, 'bright')
        elif(choose == 2):
            contrast = float(input("Enter the contrast (-255, 255): "))
            new_img = change_contrast(original_img, contrast=contrast)
            save_img(new_img, file_path, 'contrast')
        elif(choose == 3):
            print("Flipping image vertically and horizontally")
            new_img = flip_image(original_img, 1)
            save_img(new_img, file_path, 'flip_v')
            new_img = flip_image(original_img, 0)
            save_img(new_img, file_path, 'flip_h')
        elif(choose == 4):
            print("Turn image to grayscale")
            new_img = grayscale_image(img)
            save_img(new_img, file_path, 'grayscale')
        elif(choose == 5):
            print("Turn image to sepia")
            new_img = sepia_image(img)
            save_img(new_img, file_path, 'sepia')
        elif(choose == 6):
            blur_size = int(input("Enter blur size (has to be an odd number, recommend 3): "))
            sigma = int(input("Enter sigma (recommend 10): "))
            new_img = blur_image(img, blur_size, sigma)
            save_img(new_img, file_path, 'blur')
        elif(choose == 7):
            sharp_size = int(input("Enter sharp size (has to be an odd number, recommend 5): "))
            sigma = int(input("Enter sigma (recommend 10): "))
            new_img = sharp_image(img, sharp_size, sigma)
            save_img(new_img, file_path, 'sharp')
        elif(choose == 8):
            zoom = float(input("Enter how much you want to zoom in percentage: "))
            new_img = zoom_image(img, zoom);
            save_img(new_img, file_path, 'zoom')
        elif(choose == 9):
            print("Cropping image into a circle")
            new_img = crop_circle(img)
            save_img(new_img, file_path, 'circle')
        elif(choose == 0):
            print("Running all features from 1 -> 9")
            
            brightness = int(input("Enter the brightness: "))
            new_img = change_brightness(original_img, brightness=brightness)
            save_img(new_img, file_path, 'brightness')
            
            contrast = float(input("Enter the contrast (-255, 255): "))
            new_img = change_contrast(original_img, contrast=contrast)
            save_img(new_img, file_path, 'contrast')
            
            print("Flipping image vertically and horizontally")
            new_img = flip_image(original_img, 1)
            save_img(new_img, file_path, 'flip_v')
            new_img = flip_image(original_img, 0)
            save_img(new_img, file_path, 'flip_h')
            
            print("Turn image to grayscale")
            new_img = grayscale_image(original_img)
            save_img(new_img, file_path, 'grayscale')
            
            print("Turn image to sepia")
            sepia = sepia_image(original_img)
            save_img(sepia, file_path, 'sepia')
            
            blur_size = int(input("Enter blur size (has to be an odd number, recommend 3): "))
            sigma = int(input("Enter sigma (recommend 10): "))
            blurred = blur_image(original_img, blur_size, sigma)
            save_img(blurred, file_path, 'blur')
            
            sharp_size = int(input("Enter sharp size (has to be an odd number, recommend 5): "))
            sigma = int(input("Enter sigma (recommend 10): "))
            new_img = sharp_image(blurred, sharp_size, sigma)
            save_img(new_img, file_path, 'sharp')
            
            zoom = float(input("Enter how much you want to zoom in percentage: "))
            new_img = zoom_image(original_img, zoom)
            save_img(new_img, file_path, 'zoom')
            
            print("Cropping image into a circle")
            new_img = crop_circle(original_img)
            save_img(new_img, file_path, 'circle')
        
            
            
if __name__ == '__main__':
    main()