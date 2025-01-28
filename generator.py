import os
import cv2
import numpy as np
import random
import json
from pycocotools import mask

def generate_random_background(width, height):
    bg = np.full((height, width, 3), 255, dtype=np.uint8)
    for _ in range(10000): 
        color = tuple(np.random.randint(0, 255, 3).tolist())
        rect_width = random.randint(10, 50)  
        rect_height = random.randint(10, 50) 
        top_left = (random.randint(0, width - rect_width), random.randint(0, height - rect_height))
        bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)
        cv2.rectangle(bg, top_left, bottom_right, color, -1)
    return bg

def initialize_coco():
    return {
        "info": {
            "description": "Mulder Synthetic Dataset",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}],
    }

def extract_object(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_background = np.array([0, 0, 220])
    upper_background = np.array([180, 30, 255])
    background_mask = cv2.inRange(hsv, lower_background, upper_background)
    object_mask = cv2.bitwise_not(background_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
    object_mask = cv2.bitwise_and(object_mask, object_mask, mask=cv2.inRange(image, (0, 0, 0), (255, 255, 255)))
    cv2.imwrite("debug_object_mask.png", object_mask)
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return object_mask, (x, y, w, h)
    return object_mask, None

def augment_object(object_image, object_mask, bg_shape):
    h, w = object_image.shape[:2]
    expanded_canvas_size = int(1.5 * max(h, w))
    expanded_canvas = np.full((expanded_canvas_size, expanded_canvas_size, 3), 255, dtype=np.uint8)
    expanded_mask = np.zeros((expanded_canvas_size, expanded_canvas_size), dtype=np.uint8)
    y_offset = (expanded_canvas_size - h) // 2
    x_offset = (expanded_canvas_size - w) // 2
    expanded_canvas[y_offset:y_offset+h, x_offset:x_offset+w] = object_image
    expanded_mask[y_offset:y_offset+h, x_offset:x_offset+w] = object_mask
    scale = random.uniform(0.5, 1.5) 
    scaled_image = cv2.resize(expanded_canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_mask = cv2.resize(expanded_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    angle = random.uniform(-30, 30)  
    rotation_matrix = cv2.getRotationMatrix2D((scaled_image.shape[1] // 2, scaled_image.shape[0] // 2), angle, 1.0)
    rotated_image = cv2.warpAffine(scaled_image, rotation_matrix, (scaled_image.shape[1], scaled_image.shape[0]),
                                   flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    rotated_mask = cv2.warpAffine(scaled_mask, rotation_matrix, (scaled_image.shape[1], scaled_image.shape[0]),
                                  flags=cv2.INTER_NEAREST)
    non_zero_coords = np.column_stack(np.where(rotated_mask > 0))
    if non_zero_coords.size == 0:
        return None, None, 0, 0, 0, 0
    min_y, min_x = non_zero_coords.min(axis=0)
    max_y, max_x = non_zero_coords.max(axis=0)
    cropped_image = rotated_image[min_y:max_y+1, min_x:max_x+1]
    cropped_mask = rotated_mask[min_y:max_y+1, min_x:max_x+1]
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-50, 50) 
    adjusted_image = cv2.convertScaleAbs(cropped_image, alpha=alpha, beta=beta)
    if random.random() > 0.5:
        ksize = random.choice([3, 5, 7])
        adjusted_image = cv2.GaussianBlur(adjusted_image, (ksize, ksize), 0)
    if random.random() > 0.5:
        noise = np.random.normal(0, 25, adjusted_image.shape).astype(np.int16)
        adjusted_image = np.clip(adjusted_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    dx = random.randint(0, max(0, bg_shape[1] - cropped_image.shape[1]))
    dy = random.randint(0, max(0, bg_shape[0] - cropped_image.shape[0]))
    final_image = np.full(bg_shape, 255, dtype=np.uint8)
    final_mask = np.zeros((bg_shape[0], bg_shape[1]), dtype=np.uint8)
    final_image[dy:dy+adjusted_image.shape[0], dx:dx+adjusted_image.shape[1]] = adjusted_image
    final_mask[dy:dy+cropped_mask.shape[0], dx:dx+cropped_mask.shape[1]] = cropped_mask
    return final_image, final_mask, dx, dy, cropped_image.shape[1], cropped_image.shape[0]

def process_images(input_folder, output_folder, coco_file, count):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    coco = initialize_coco()
    annotation_id = 1
    image_id = 1
    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, file_name)
            image = cv2.imread(image_path)
            obj_mask, bbox = extract_object(image)
            if bbox is None:
                continue
            x, y, w, h = bbox
            object_roi = image[y:y+h, x:x+w]
            mask_cropped = obj_mask[y:y+h, x:x+w]
            object_with_white = cv2.bitwise_and(object_roi, object_roi, mask=mask_cropped)
            for _ in range(count): 
                bg = generate_random_background(image.shape[1], image.shape[0])
                augmented_image, augmented_mask, dx, dy, obj_w, obj_h = augment_object(object_with_white, mask_cropped, bg.shape)
                if augmented_image is None or augmented_mask is None:
                    continue
                mask_3d = cv2.merge([augmented_mask] * 3)
                inv_mask = cv2.bitwise_not(mask_3d)
                bg_with_object = cv2.bitwise_and(bg, inv_mask) + cv2.bitwise_and(augmented_image, mask_3d)
                new_file_name = f"synthetic_{image_id}.jpg"
                new_image_path = os.path.join(output_folder, new_file_name)
                cv2.imwrite(new_image_path, bg_with_object)
                coco["images"].append({
                    "id": image_id,
                    "file_name": new_file_name,
                    "width": image.shape[1],
                    "height": image.shape[0],
                })
                rle = mask.encode(np.asfortranarray(augmented_mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [dx, dy, obj_w, obj_h],
                    "area": obj_w * obj_h,
                    "segmentation": rle,
                    "iscrowd": 0,
                })
                annotation_id += 1
                image_id += 1
    with open(coco_file, "w") as f:
        json.dump(coco, f, indent=4)

input_folder = "./images"
output_folder = "./o_images"
coco_file = "./annotations.json"
process_images(input_folder, output_folder, coco_file, count=20)
