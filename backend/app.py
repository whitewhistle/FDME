from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
from flask_cors import CORS
from utils import convert_bgr_to_rgb,convert_rgb_to_gray,apply_canny_edge_detection,snic_superpixel_image,estimate_sparse_blur,slic_superpixel,process_image_slic,process_image_snic
import numpy as np
app = Flask(__name__)
CORS (app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

# Ensure the folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    std1 = float(request.form.get('std1', 1))  # Default to 1 if not provided
    std2 = float(request.form.get('std2', 2))  # Default to 2 if not provided
    n_superpixels = int(request.form.get('nSuperpixels', 200))
    if std2 <= std1:
        return jsonify({"error": "std2 must be greater than std1"}), 400
    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Process the image (e.g., SLIC segmentation)
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({"error": "Invalid image format"}), 400
    rgb_image = convert_bgr_to_rgb(image)
    gray_image = convert_rgb_to_gray(rgb_image)
    canny_image = apply_canny_edge_detection(gray_image)
    sparse_bmap,mag1,mag2 = estimate_sparse_blur(gray_image, canny_image, std1=1, std2=2)  
    seed_mask = sparse_bmap > 0
    slic_image = slic_superpixel (rgb_image,n_superpixels,10)
    dense_defocus_slic, superpixels = process_image_slic (
    rgb_image, 
    sparse_bmap, 
    seed_mask,
    n_segments=n_superpixels,
    compactness=20,
)
    snic_image = snic_superpixel_image(rgb_image,n_superpixels,10)
    dense_defocus_snic, superpixels_snic = process_image_snic(
    rgb_image, 
    sparse_bmap, 
    seed_mask,
    n_segments=n_superpixels,
    compactness=20,
)
    rgb = os.path.join(UPLOAD_FOLDER, 'converted_rgb.png')
    gray = os.path.join(UPLOAD_FOLDER, 'grayscale_image.png')
    canny = os.path.join(UPLOAD_FOLDER, 'canny_image.png')
    sparse_p = os.path.join(UPLOAD_FOLDER, 'sparse.png')
    mag1_p = os.path.join(UPLOAD_FOLDER, 'mag1.png')
    mag2_p = os.path.join(UPLOAD_FOLDER, 'mag2.png')
    slic_p = os.path.join(UPLOAD_FOLDER, 'slic.png')
    dense_slic_p = os.path.join(UPLOAD_FOLDER, 'dense_slic.png')
    snic_p = os.path.join(UPLOAD_FOLDER, 'snic.png')
    dense_snic_p = os.path.join(UPLOAD_FOLDER, 'dense_snic.png')
    plt.imsave(rgb, rgb_image)
    plt.imsave(gray, gray_image, cmap='gray')
    plt.imsave(canny, canny_image, cmap='gray')
    plt.imsave(sparse_p, sparse_bmap, cmap='hot')
    plt.imsave(mag1_p, mag1, cmap='hot')
    plt.imsave(mag2_p, mag2, cmap='hot')
    plt.imsave(slic_p, slic_image)
    plt.imsave(dense_slic_p, dense_defocus_slic,cmap='gray')
    plt.imsave(snic_p, snic_image)
    plt.imsave(dense_snic_p, dense_defocus_snic,cmap='gray')
    return jsonify({
        "rgb_image_url": f'/uploads/converted_rgb.png',
        "gray_image_url": f'/uploads/grayscale_image.png',
        "canny_image_url": f'/uploads/canny_image.png',
        "sparse_image_url": f'/uploads/sparse.png',
        "mag1_image_url": f'/uploads/mag1.png',
        "mag2_image_url": f'/uploads/mag2.png',
        "slic_image_url": f'/uploads/slic.png',
        "dense_slic_image_url": f'/uploads/dense_slic.png',
        "snic_image_url": f'/uploads/snic.png',
        "dense_snic_image_url": f'/uploads/dense_snic.png',
    }), 200

@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
