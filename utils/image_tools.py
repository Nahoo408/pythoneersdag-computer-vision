import cv2
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np
from ipywidgets import interact, IntSlider

def draw_rectangle(
        image: NDArray,
        rectangle: tuple[int, int, int, int],
        colour: tuple[int, int, int] = (255, 0, 0),
        line_width: int = 2,
) -> None:
    """Draw given rectangle on given image.

    The rectangle is expected to be of format (x_center, y_center, width, height),
    all in pixels.

    """
    x_center, y_center, width, height = rectangle
    cv2.rectangle(image, (x_center, y_center), (x_center + width, y_center + height), colour, line_width)


def interactive_hsv_tuner(img: NDArray, resize_width: int = 800):
    """
    Launches an interactive HSV tuner with sliders for an input image.

    Parameters:
    - image_path (str): Path to the image.
    - resize_width (int): Width to resize the image for display.
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (resize_width, int(img.shape[0] * resize_width / img.shape[1])))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)

    # Slider callback function
    def hsv_filter(h_low, h_high, s_low, s_high, v_low, v_high):
        lower = np.array([h_low, s_low, v_low])
        upper = np.array([h_high, s_high, v_high])

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(img_resized, img_resized, mask=mask)

        # Show side-by-side
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].imshow(img_resized)
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        ax[1].imshow(result)
        ax[1].set_title("HSV Filtered")
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()

    # Create interactive sliders
    interact(
        hsv_filter,
        h_low=IntSlider(0, 0, 179, description='H Low'),
        h_high=IntSlider(179, 0, 179, description='H High'),
        s_low=IntSlider(0, 0, 255, description='S Low'),
        s_high=IntSlider(255, 0, 255, description='S High'),
        v_low=IntSlider(0, 0, 255, description='V Low'),
        v_high=IntSlider(255, 0, 255, description='V High'),
    )

def interactive_rgb_tuner(img: NDArray, resize_width: int = 800):
    """
    Launches an interactive RGB tuner with sliders for an input image.

    Parameters:
    - image_path (str): Path to the image.
    - resize_width (int): Width to resize the image for display.
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (resize_width, int(img.shape[0] * resize_width / img.shape[1])))

    # Slider callback function
    def rgb_filter(r_low, r_high, g_low, g_high, b_low, b_high):
        lower = np.array([r_low, g_low, b_low])
        upper = np.array([r_high, g_high, b_high])

        # Create mask and apply
        mask = cv2.inRange(img_resized, lower, upper)
        result = cv2.bitwise_and(img_resized, img_resized, mask=mask)

        # Show side-by-side
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].imshow(img_resized)
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        ax[1].imshow(result)
        ax[1].set_title("RGB Filtered")
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()

    # Create interactive sliders
    interact(
        rgb_filter,
        r_low=IntSlider(0, 0, 255, description='R Low'),
        r_high=IntSlider(255, 0, 255, description='R High'),
        g_low=IntSlider(0, 0, 255, description='G Low'),
        g_high=IntSlider(255, 0, 255, description='G High'),
        b_low=IntSlider(0, 0, 255, description='B Low'),
        b_high=IntSlider(255, 0, 255, description='B High'),
    )

def nothing(_: int) -> None:
    """
    Dummy callback function used for OpenCV trackbars.

    Args:
        _ (int): The current value of the trackbar (automatically passed by OpenCV).

    This function does nothing and is typically used as a placeholder when a callback is required.
    """
    pass

def read_image(path: str) -> NDArray:
    """
    Read an image from the given file path.

    Args:
        path (str): Path to the image file.

    Returns:
        NDArray: Loaded image in BGR format.
    """
    return cv2.imread(path)

def rgb_to_grayscale(image: NDArray) -> NDArray:
    """
    Convert a BGR image to grayscale.

    Args:
        image (NDArray): Input image in BGR format.

    Returns:
        NDArray: Grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def show_img_inline(image: NDArray, title: str = "", **kwargs) -> None:
    """
    Show an image inline in Jupyter Notebook using matplotlib.

    Args:
        image (NDArray): Image to display (BGR format).
        title (str, optional): Title for the image. Defaults to "".
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), **kwargs)
    plt.title(title)
    plt.axis("off")
    plt.show()