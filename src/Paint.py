import cv2
import numpy as np

class Paint:

    @staticmethod
    def match_histogram(content_path, styled_path, output_path):
        # Transfers the color palette from styled_path to content_path
        content = cv2.imread(content_path)
        styled = cv2.imread(styled_path)

        # Convert images to LAB (separate brightness and colors)
        content_lab = cv2.cvtColor(content, cv2.COLOR_BGR2LAB)
        styled_lab = cv2.cvtColor(styled, cv2.COLOR_BGR2LAB)

        # Calculate mean and standard deviation for A and B channels
        c_mean, c_std = cv2.meanStdDev(content_lab)
        s_mean, s_std = cv2.meanStdDev(styled_lab)
        s_std = np.maximum(s_std, 1e-6)

        # Apply color transformation (replace A and B channels, keep L unchanged)
        a_new = (content_lab[..., 1] - c_mean[1]) * (s_std[1] / c_std[1]) + s_mean[1]
        b_new = (content_lab[..., 2] - c_mean[2]) * (s_std[2] / c_std[2]) + s_mean[2]

        # Clip values to avoid going out of range 0-255
        a_new, b_new = [np.clip(channel, 0, 255).astype(np.uint8) for channel in (a_new, b_new)]

        # Assemble the final image (brightness from content, colors from style)
        result_lab = cv2.merge([content_lab[..., 0], a_new, b_new])
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

        cv2.imwrite(output_path, result)

    @staticmethod
    def sharpen_image(image_path, output_path):
        # Applies a sharpening filter to image_path
        image = cv2.imread(image_path)
        kernel = np.array([[0, -1, 0],
                           [-1,  5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        cv2.imwrite(output_path, sharpened)

    @staticmethod
    def enhance_colors(image_path, output_path, strength=1.4):
        # Enhances the color saturation of image_path by a given strength factor
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = cv2.multiply(hsv[..., 1], strength)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(output_path, result)
