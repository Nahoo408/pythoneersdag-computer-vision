import cv2
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def read_image(path: str) -> NDArray:
    """
    Read an image from the given file path.

    Args:
        path (str): Path to the image file.

    Returns:
        NDArray: Loaded image in BGR format.
    """
    return cv2.imread(path)


def show_image(image: NDArray) -> None:
    """
    Show an image in a separate OpenCV window. Press Esc to close.

    Args:
        image (NDArray): Image to display.
    """
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_img_inline(image: NDArray, title: str = "", cmap=None) -> None:
    """
    Show an image inline in Jupyter Notebook using matplotlib.

    Args:
        image (NDArray): Image to display (BGR format).
        title (str, optional): Title for the image. Defaults to "".
        cmap (str, optional): Colormap to use for grayscale. Defaults to None.
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


def rgb_to_grayscale(image: NDArray) -> NDArray:
    """
    Convert a BGR image to grayscale.

    Args:
        image (NDArray): Input image in BGR format.

    Returns:
        NDArray: Grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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


def draw_rectangle_xyxy(
    image: NDArray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    colour: tuple[int, int, int] = (0, 255, 0),
    line_width: int = 2,
) -> NDArray:
    """
    Draw a rectangle given corner coordinates.

    Args:
        image (NDArray): Image to draw on.
        x1 (int): Top-left x-coordinate.
        y1 (int): Top-left y-coordinate.
        x2 (int): Bottom-right x-coordinate.
        y2 (int): Bottom-right y-coordinate.
        colour (tuple[int, int, int], optional): Rectangle color (BGR). Defaults to green.
        line_width (int, optional): Line thickness. Defaults to 2.

    Returns:
        NDArray: New image with rectangle drawn.
    """
    img_copy = image.copy()
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), colour, line_width)
    return img_copy


def xyxy_to_xywh(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    """
    Convert coordinates from (x1, y1, x2, y2) to (x, y, width, height).

    Args:
        x1 (int): Top-left x-coordinate.
        y1 (int): Top-left y-coordinate.
        x2 (int): Bottom-right x-coordinate.
        y2 (int): Bottom-right y-coordinate.

    Returns:
        tuple[int, int, int, int]: Rectangle in (x, y, width, height) format.
    """
    width = x2 - x1
    height = y2 - y1
    return x1, y1, width, height
