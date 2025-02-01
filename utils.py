import os
import io
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdf2image
from PIL import Image
from PIL.PpmImagePlugin import PpmImageFile
import pytesseract

import pymupdf as fitz
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import SORT_MODE_BASIC, SORT_MODE_XY_CUT
from unstructured.partition.utils.xycut import (
    bbox2points,
    recursive_xy_cut,
    vis_polygons_with_index,
)


##---------------------------------------------------------------------------------------------------------------------------------------------------

def is_page_scanned(page, text_threshold=0.1):
    contains_text = False
    contains_image = False

    # Get the extracted text and measure its length
    text = page.get_text("text").strip()
    text_length = len(text)

    # Get the dimensions of the page (width * height) to determine the total page area
    page_width, page_height = page.rect.width, page.rect.height
    page_area = page_width * page_height

    # Check if there's any text at all
    if text_length > 0:
        contains_text = True

    # Get the text bounding boxes
    text_blocks = page.get_text("dict")["blocks"]

    # Calculate the total area of all text blocks
    total_text_area = 0
    for block in text_blocks:
        if block["type"] != 0:  # type for text = 0 and for image = 1
            continue  # Skip non-text blocks
        if "bbox" in block:  # Only consider blocks that have bounding boxes (text blocks)
            bbox = block["bbox"]  # Bounding box is a tuple (x0, y0, x1, y1)
            block_width = bbox[2] - bbox[0]  # x1-x0
            block_height = bbox[3] - bbox[1] # y1-y0
            block_area = block_width * block_height
            total_text_area += block_area

    # Check if there's any text at all
    if total_text_area > 0:
        contains_text = True

    # Calculate the text density as the proportion of text area to page area
    if page_area > 0:
        text_density = total_text_area / page_area
    else:
        text_density = 0

    # Check if the page contains images
    if page.get_images(full=True):
        contains_image = True

    # print("Total page area =", page_area)
    # print("Total text area =", total_text_area)
    # print("Text density", text_density)

    # Apply the threshold: if the text density is below the threshold, consider it scanned
    return (contains_image and text_density < text_threshold) or (not contains_text and contains_image) or (not contains_text)


def is_page_scanned(page, text_threshold=0.05, image_threshold=0.5):
    # Get the dimensions of the page (width * height) to determine the total page area
    page_width, page_height = page.rect.width, page.rect.height
    page_area = page_width * page_height

    # Get the text bounding boxes
    blocks = page.get_text("dict")["blocks"]

    # Calculate the total area of all text blocks
    total_text_area = 0
    total_img_area = 0
    for block in blocks:
        if (block["type"] == 0)  and ("bbox" in block):  # Only consider blocks that have bounding boxes (text blocks)
            bbox = block["bbox"]  # Bounding box is a tuple (x0, y0, x1, y1)
            block_width = bbox[2] - bbox[0]  # x1-x0
            block_height = bbox[3] - bbox[1] # y1-y0
            block_area = block_width * block_height
            total_text_area += block_area

        elif block["type"] == 1:
            img_width, img_height = block["width"], block["height"]
            img_area = img_width * img_height
            total_img_area += img_area

        else:
            continue

    # Check if there's any text at all
    if total_text_area > 0:
        contains_text = True

    if total_img_area > 0:
        contains_image = True

    # Calculate the text density as the proportion of text area to page area
    if page_area > 0:
        text_density = total_text_area / page_area
        image_density = total_img_area / page_area
    else:
        text_density = 0
        image_density = 0

    print(f"Text density = {text_density}")
    print(f"Image density = {image_density}")

    if image_density == 1.0:
        return True
    elif text_density == 1.0:
        return False
    elif text_density < text_threshold and image_density > image_threshold:
        return True
    else:
        return False


def is_pdf_scanned(pdf_path):
    """
    Detect if a PDF is scanned or editable by checking for text or images in the PDF.
    """
    # Open the PDF document
    doc = fitz.open(pdf_path)
    pages_scanned_status = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Check if the page is scanned
        if is_page_scanned(page):
            pages_scanned_status.append(1)
        else:
            pages_scanned_status.append(0)

    # Check if all pages are scanned
    if all(pages_scanned_status):
        return "scanned", pages_scanned_status
    # Check if any page is editable
    elif any(pages_scanned_status):
        return "mixed", pages_scanned_status
    else:
        return "editable", pages_scanned_status



def preprocess_image(image: np.ndarray):
    """Preprocess the scanned image to enhance text areas."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

     # Apply Non-Local Means Denoising to reduce noise
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    return contrast_enhanced

def detect_orientation(image: np.ndarray):
    """Detect the text orientation in a scanned image using Tesseract."""
    # Preprocess the image (if needed)
    processed_image = preprocess_image(image)

    # Use Tesseract to detect the orientation and script direction
    config = "--psm 0"  # Page segmentation mode 0 detects orientation and script
    ocr_result = pytesseract.image_to_osd(processed_image, config=config)

    # Extract the rotation angle from the OSD output
    orientation_angle = int(ocr_result.split("Orientation in degrees:")[1].split("\n")[0].strip())
    rotation_angle = int(ocr_result.split("Rotate:")[1].split("\n")[0].strip())

    return orientation_angle, rotation_angle


def rotate_image(image: np.ndarray, angle: int):
    """Rotate the image to correct its orientation."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix for rotating the image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the absolute cosine and sine of the rotation angle
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    # Compute the new bounding box dimensions
    new_w = int((h * abs_sin) + (w * abs_cos))
    new_h = int((h * abs_cos) + (w * abs_sin))

    # # Adjust the rotation matrix to account for the translation
    # M[0, 2] += (new_w / 2) - center[0]
    # M[1, 2] += (new_h / 2) - center[1]

    # Adjust the translation part of the rotation matrix to ensure centering
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

    return rotated_image


def align_image(img: PpmImageFile):
    # conert PIL image to cv image
    img = np.array(img)
    # Convert RGB to BGR (if necessary)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    skew_angle, rotation_angle = detect_orientation(img)
    img = rotate_image(img, -rotation_angle)
    return img


def convert_images_to_temp_pdf(image_list: list, op_file_path: str):
    """
    Convert a list of images to a PDF and save it in a temporary file.
    :param image_list: List of image file paths or PIL Image objects.
    :return: The temporary file object containing the PDF (context-managed).
    """
    images = []

    for image in image_list:
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image  # If image is already a PIL Image object

        # Convert image to RGB (if it's not already)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        images.append(img)

    # Save the images as a PDF (first image + rest of the images)
    if images:
        images[0].save(op_file_path, save_all=True, append_images=images[1:], resolution=100.0)


def partition_scanned_pdf(pdf_path: str):
    images = pdf2image.convert_from_path(pdf_path)

    for i in range(len(images)):
        images[i] = align_image(images[i])

    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
        temp_pdf_path = temp_pdf.name
        convert_images_to_temp_pdf(images, temp_pdf_path)
        print(f"Temporary PDF file created: {temp_pdf_path}")
        elements = partition_pdf(temp_pdf_path,
                                strategy="hi_res",
                                content_type = "application/pdf",
                                # chunking_strategy = "by_title",
                                # include_page_breaks=True,
                                sort_mode=SORT_MODE_XY_CUT)

    return elements


def pil_image_to_bytes(pil_image, image_format='PNG'):
    """Convert a PIL image to bytes."""
    # Create a BytesIO object to hold the image data
    img_byte_arr = io.BytesIO()

    # Save the image to the BytesIO object in the specified format
    pil_image.save(img_byte_arr, format=image_format)

    # Get the byte data
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr


def partition_scanned_pdf_by_images(pdf_path: str):
    images = pdf2image.convert_from_path(pdf_path)
    elements = []
    for image in images:
        image = align_image(image)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # use BytesIO to create a file-like object
        with io.BytesIO() as temp_file:
            image.save(temp_file, format='PNG')  # Or the appropriate format
            temp_file.seek(0)  # Reset the file pointer to the beginning
            element = partition_image(file=temp_file,
                                      strategy="hi_res",
                                      content_type="image/png",
                                      chunking_strategy = "by_title",
                                      sort_mode=SORT_MODE_XY_CUT)
            elements.extend(element)
    return elements


##---------------------------------------------------------------------------------------------------------------------------------------------------

def detect_skew_angle(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to smooth the image (reduce noise)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize the image (apply threshold)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use morphological operations to enhance the text (close gaps between characters)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))  # A larger kernel for horizontal lines
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(morphed, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=5)

    if lines is None:
        # If no lines are detected, return a skew angle of 0
        return 0

    # Calculate the angles of the detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Filter out horizontal lines (those with angles close to 0 or 180)
    filtered_angles = [angle for angle in angles if abs(angle) > 1.0 and abs(angle) < 89.0]

    if len(filtered_angles) == 0:
        # If no valid angles are detected, return 0
        return 0

    # Return the median of the filtered angles (more robust to noise)
    median_angle = np.median(filtered_angles)

    # Normalize the angle to be within [-180, 180)
    if median_angle < -180:
        median_angle += 360
    elif median_angle > 180:
        median_angle -= 360

    return median_angle


def detect_nearest_angle(estimated_angle):
    """Round the detected angle to the nearest of -180°, -90°, 0°, 90°, or 180°."""
    possible_angles = [-180, -90, 0, 90, 180]

    # Normalize the angle to be within [-180, 180)
    estimated_angle = (estimated_angle + 180) % 360 - 180

    # Find the closest angle
    nearest_angle = min(possible_angles, key=lambda x: abs(estimated_angle - x))

    return nearest_angle

# Rotate the image to correct the skew
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # Get the rotation matrix for correcting skew
    rotation_matrix = cv2.getRotationMatrix2D(center = center, angle = angle, scale = 1.0)
    # Rotate the image
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return rotated


##--------------------------------------------------------------------------------------------------------------------------------------------------------------
def detect_skew_angle_v1(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to smooth the image (reduce noise)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize the image (apply threshold)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use morphological operations to enhance the text (close gaps between characters)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))  # A larger kernel for horizontal lines
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(morphed, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=5)

    if lines is None:
        # If no lines are detected, return a skew angle of 0
        return 0

    # Calculate the angles of the detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Filter out horizontal lines (those with angles close to 0 or 180)
    filtered_angles = [angle for angle in angles if abs(angle) > 1.0 and abs(angle) < 89.0]

    if len(filtered_angles) == 0:
        # If no valid angles are detected, return 0
        return 0

    # Return the median of the filtered angles (more robust to noise)
    median_angle = np.median(filtered_angles)

    # Normalize the angle to be within [0, 360)
    if median_angle < 0:
        median_angle += 360

    return median_angle


def detect_skew_angle_v2(image):
    """Detect the skew angle of a scanned PDF page."""
    # Preprocess image (binary and morphological operations)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

     # Apply Non-Local Means Denoising to reduce noise
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(contrast_enhanced, 50, 150, apertureSize=3)

    # Detect lines in the image using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return 0  # No skew detected, assume it's already aligned

    # Calculate angles of all detected lines
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90  # Convert radians to degrees
        if angle > -45 and angle < 45:
            angles.append(angle)

    # Compute the median angle
    if len(angles) > 0:
        skew_angle = np.median(angles)
        return skew_angle
    else:
        return 0  # Assume no skew if no valid angles found


def detect_nearest_angle(estimated_angle):
    """Round the detected angle to the nearest of 0°, 90°, 180°, or 270°."""
    possible_angles = [0, 90, 180, 270, 360]

    # Normalize the angle to be within [0, 360)
    estimated_angle = estimated_angle % 360

    # Find the closest angle
    nearest_angle = min(possible_angles, key=lambda x: abs(estimated_angle - x))

    # If the nearest angle is 360, treat it as 0
    if nearest_angle == 360:
        nearest_angle = 0

    return nearest_angle

# Rotate the image to correct the skew
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # Get the rotation matrix for correcting skew
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Rotate the image
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return rotated


##--------------------------------------------------------------------------------------------------------------------------------------------------------------

def detect_skew_angle(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to smooth the image (reduce noise)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize the image (apply threshold)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use morphological operations to enhance the text (close gaps between characters)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))  # A larger kernel for horizontal lines
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(morphed, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=5)

    if lines is None:
        # If no lines are detected, return a skew angle of 0
        return 0

    # Calculate the angles of the detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Filter out horizontal lines (those with angles close to 0 or 180)
    filtered_angles = [angle for angle in angles if abs(angle) > 1.0 and abs(angle) < 89.0]

    if len(filtered_angles) == 0:
        # If no valid angles are detected, return 0
        return 0

    # Return the median of the filtered angles (more robust to noise)
    median_angle = np.median(filtered_angles)

    # Normalize the angle to be within [-180, 180)
    if median_angle < -180:
        median_angle += 360
    elif median_angle > 180:
        median_angle -= 360

    return median_angle


def detect_nearest_angle(estimated_angle):
    """Round the detected angle to the nearest of -180°, -90°, 0°, 90°, or 180°."""
    possible_angles = [-180, -90, 0, 90, 180]

    # Normalize the angle to be within [-180, 180)
    estimated_angle = (estimated_angle + 180) % 360 - 180

    # Find the closest angle
    nearest_angle = min(possible_angles, key=lambda x: abs(estimated_angle - x))

    return nearest_angle

# Rotate the image to correct the skew
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # Get the rotation matrix for correcting skew
    rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
    # Rotate the image
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return rotated


##--------------------------------------------------------------------------------------------------------------------------------------------------------------

def show_plot(image, desired_width=None):
    image_height, image_width, _ = image.shape
    if desired_width:
        # Calculate the desired height based on the original aspect ratio
        aspect_ratio = image_width / image_height
        desired_height = desired_width / aspect_ratio

        # Create a figure with the desired size and aspect ratio
        fig, ax = plt.subplots(figsize=(desired_width, desired_height))
    else:
        # Create figure and axes
        fig, ax = plt.subplots()
    # Display the image
    ax.imshow(image)
    plt.show()


def extract_element_coordinates(elements):
    elements_coordinates = []
    page_elements_coordinates = []

    for el in elements:
        if isinstance(el, us.documents.elements.PageBreak):
            if page_elements_coordinates:
                elements_coordinates.append(page_elements_coordinates)
                page_elements_coordinates = []
        else:
            page_elements_coordinates.append(el.metadata.coordinates)

    if page_elements_coordinates:
        elements_coordinates.append(page_elements_coordinates)

    return elements_coordinates


def convert_coordinates_to_boxes(coordinates, image):
    boxes = []

    for coordinate in coordinates:
        points = coordinate.points
        _left, _top = points[0]
        _right, _bottom = points[2]
        w = coordinate.system.width
        h = coordinate.system.height
        image_height, image_width, _ = image.shape
        left = _left * image_width / w
        right = _right * image_width / w
        top = _top * image_height / h
        bottom = _bottom * image_height / h
        boxes.append([int(left), int(top), int(right), int(bottom)])

    return boxes


def order_boxes(boxes):
    res = []
    recursive_xy_cut(np.asarray(boxes).astype(int), np.arange(len(boxes)), res)
    np_array_boxes = np.array(boxes)
    ordered_boxes = np_array_boxes[np.array(res)].tolist()
    return ordered_boxes


def draw_boxes(image, boxes, output_dir, base_name, page_num, output_type, label):
    annotated_image = vis_polygons_with_index(image, [bbox2points(it) for it in boxes])

    if output_type in ["plot", "all"]:
        print(f"{label} elements - Page: {page_num}")
        show_plot(annotated_image, desired_width=20)

    if output_type in ["image", "all"]:
        output_image_path = os.path.join(output_dir, f"{base_name}_{page_num}_{label}.jpg")
        cv2.imwrite(output_image_path, annotated_image)


def draw_elements(elements, images, output_type, output_dir, base_name, label):
    elements_coordinates = extract_element_coordinates(elements)

    assert len(images) == len(elements_coordinates)
    for idx, (img, coords_per_page) in enumerate(zip(images, elements_coordinates)):
        image = np.array(img)
        boxes = convert_coordinates_to_boxes(coords_per_page, image)
        draw_boxes(image, boxes, output_dir, base_name, idx + 1, output_type, label)


def run_partition_pdf(
    pdf_path,
    strategy,
    images,
    output_type="plot",
    output_root_dir="",
):
    print(f">>> Starting run_partition_pdf - f_path: {pdf_path} - strategy: {strategy}")
    f_base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    output_dir = os.path.join(output_root_dir, strategy, f_base_name)
    os.makedirs(output_dir, exist_ok=True)

    original_elements = partition_pdf(
        filename=pdf_path,
        strategy=strategy,
        include_page_breaks=True,
        sort_mode=SORT_MODE_BASIC,
    )
    draw_elements(original_elements, images, output_type, output_dir, f_base_name, "original")

    ordered_elements = partition_pdf(
        filename=pdf_path,
        strategy=strategy,
        include_page_breaks=True,
        sort_mode=SORT_MODE_XY_CUT,
    )
    draw_elements(ordered_elements, images, output_type, output_dir, f_base_name, "result")
    print("<<< Finished run_partition_pdf")


##-------------------------------------------------------------------------------------------------------------------------------------------------------

def column_boxes(page, footer_margin=50, header_margin=50, no_image_text=True):
    """Determine bboxes which wrap a column."""
    paths = page.get_drawings()
    bboxes = []

    # path rectangles
    path_rects = []

    # image bboxes
    img_bboxes = []

    # bboxes of non-horizontal text
    # avoid when expanding horizontal text boxes
    vert_bboxes = []

    # compute relevant page area
    clip = +page.rect
    clip.y1 -= footer_margin  # Remove footer area
    clip.y0 += header_margin  # Remove header area

    def can_extend(temp, bb, bboxlist):
        """Determines whether rectangle 'temp' can be extended by 'bb'
        without intersecting any of the rectangles contained in 'bboxlist'.

        Items of bboxlist may be None if they have been removed.

        Returns:
            True if 'temp' has no intersections with items of 'bboxlist'.
        """
        for b in bboxlist:
            if not intersects_bboxes(temp, vert_bboxes) and (
                b == None or b == bb or (temp & b).is_empty
            ):
                continue
            return False

        return True

    def in_bbox(bb, bboxes):
        """Return 1-based number if a bbox contains bb, else return 0."""
        for i, bbox in enumerate(bboxes):
            if bb in bbox:
                return i + 1
        return 0

    def intersects_bboxes(bb, bboxes):
        """Return True if a bbox intersects bb, else return False."""
        for bbox in bboxes:
            if not (bb & bbox).is_empty:
                return True
        return False

    def extend_right(bboxes, width, path_bboxes, vert_bboxes, img_bboxes):
        """Extend a bbox to the right page border.

        Whenever there is no text to the right of a bbox, enlarge it up
        to the right page border.

        Args:
            bboxes: (list[IRect]) bboxes to check
            width: (int) page width
            path_bboxes: (list[IRect]) bboxes with a background color
            vert_bboxes: (list[IRect]) bboxes with vertical text
            img_bboxes: (list[IRect]) bboxes of images
        Returns:
            Potentially modified bboxes.
        """
        for i, bb in enumerate(bboxes):
            # do not extend text with background color
            if in_bbox(bb, path_bboxes):
                continue

            # do not extend text in images
            if in_bbox(bb, img_bboxes):
                continue

            # temp extends bb to the right page border
            temp = +bb
            temp.x1 = width

            # do not cut through colored background or images
            if intersects_bboxes(temp, path_bboxes + vert_bboxes + img_bboxes):
                continue

            # also, do not intersect other text bboxes
            check = can_extend(temp, bb, bboxes)
            if check:
                bboxes[i] = temp  # replace with enlarged bbox

        return [b for b in bboxes if b != None]

    def clean_nblocks(nblocks):
        """Do some elementary cleaning."""

        # 1. remove any duplicate blocks.
        blen = len(nblocks)
        if blen < 2:
            return nblocks
        start = blen - 1
        for i in range(start, -1, -1):
            bb1 = nblocks[i]
            bb0 = nblocks[i - 1]
            if bb0 == bb1:
                del nblocks[i]

        # 2. repair sequence in special cases:
        # consecutive bboxes with almost same bottom value are sorted ascending
        # by x-coordinate.
        y1 = nblocks[0].y1  # first bottom coordinate
        i0 = 0  # its index
        i1 = -1  # index of last bbox with same bottom

        # Iterate over bboxes, identifying segments with approx. same bottom value.
        # Replace every segment by its sorted version.
        for i in range(1, len(nblocks)):
            b1 = nblocks[i]
            if abs(b1.y1 - y1) > 10:  # different bottom
                if i1 > i0:  # segment length > 1? Sort it!
                    nblocks[i0 : i1 + 1] = sorted(
                        nblocks[i0 : i1 + 1], key=lambda b: b.x0
                    )
                y1 = b1.y1  # store new bottom value
                i0 = i  # store its start index
            i1 = i  # store current index
        if i1 > i0:  # segment waiting to be sorted
            nblocks[i0 : i1 + 1] = sorted(nblocks[i0 : i1 + 1], key=lambda b: b.x0)
        return nblocks

    # extract vector graphics
    for p in paths:
        path_rects.append(p["rect"].irect)
    path_bboxes = path_rects

    # sort path bboxes by ascending top, then left coordinates
    path_bboxes.sort(key=lambda b: (b.y0, b.x0))

    # bboxes of images on page, no need to sort them
    for item in page.get_images():
        img_bboxes.extend(page.get_image_rects(item[0]))

    # blocks of text on page
    blocks = page.get_text(
        "dict",
        flags=fitz.TEXTFLAGS_TEXT,
        clip=clip,
    )["blocks"]

    # Make block rectangles, ignoring non-horizontal text
    for b in blocks:
        bbox = fitz.IRect(b["bbox"])  # bbox of the block

        # ignore text written upon images
        if no_image_text and in_bbox(bbox, img_bboxes):
            continue

        # confirm first line to be horizontal
        line0 = b["lines"][0]  # get first line
        if line0["dir"] != (1, 0):  # only accept horizontal text
            vert_bboxes.append(bbox)
            continue

        srect = fitz.EMPTY_IRECT()
        for line in b["lines"]:
            lbbox = fitz.IRect(line["bbox"])
            text = "".join([s["text"].strip() for s in line["spans"]])
            if len(text) > 1:
                srect |= lbbox
        bbox = +srect

        if not bbox.is_empty:
            bboxes.append(bbox)

    # Sort text bboxes by ascending background, top, then left coordinates
    bboxes.sort(key=lambda k: (in_bbox(k, path_bboxes), k.y0, k.x0))

    # Extend bboxes to the right where possible
    bboxes = extend_right(
        bboxes, int(page.rect.width), path_bboxes, vert_bboxes, img_bboxes
    )

    # immediately return of no text found
    if bboxes == []:
        return []

    # --------------------------------------------------------------------
    # Join bboxes to establish some column structure
    # --------------------------------------------------------------------
    # the final block bboxes on page
    nblocks = [bboxes[0]]  # pre-fill with first bbox
    bboxes = bboxes[1:]  # remaining old bboxes

    for i, bb in enumerate(bboxes):  # iterate old bboxes
        check = False  # indicates unwanted joins

        # check if bb can extend one of the new blocks
        for j in range(len(nblocks)):
            nbb = nblocks[j]  # a new block

            # never join across columns
            if bb == None or nbb.x1 < bb.x0 or bb.x1 < nbb.x0:
                continue

            # never join across different background colors
            if in_bbox(nbb, path_bboxes) != in_bbox(bb, path_bboxes):
                continue

            temp = bb | nbb  # temporary extension of new block
            check = can_extend(temp, nbb, nblocks)
            if check == True:
                break

        if not check:  # bb cannot be used to extend any of the new bboxes
            nblocks.append(bb)  # so add it to the list
            j = len(nblocks) - 1  # index of it
            temp = nblocks[j]  # new bbox added

        # check if some remaining bbox is contained in temp
        check = can_extend(temp, bb, bboxes)
        if check == False:
            nblocks.append(bb)
        else:
            nblocks[j] = temp
        bboxes[i] = None

    # do some elementary cleaning
    nblocks = clean_nblocks(nblocks)

    # return identified text bboxes
    return nblocks

def main():
    sample_pdf_path = ""
    doc = fitz.open(sample_pdf_path)

    for page in doc:
        bboxes = column_boxes(page, footer_margin=25, no_image_text=True)
        for rect in bboxes:
            print(page.get_text(clip=rect, sort=True))
        print("-" * 80)
