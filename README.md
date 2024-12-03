# Image Processing with Flask and React

This project is a web-based image processing application that leverages Flask for backend image processing and React for frontend user interaction. The application allows users to upload images, configure processing parameters, and view results for various image processing techniques such as SLIC and SNIC superpixel segmentation, grayscale conversion, edge detection, and defocus blur estimation.

---

## Features

- Upload images through a React-based frontend.
- Process images using advanced techniques:
  - SLIC Superpixel Segmentation
  - SNIC Superpixel Segmentation
  - Grayscale Conversion
  - Canny Edge Detection
  - Sparse Blur Estimation
  - Dense Defocus Blur Estimation
- Dynamic parameter configuration for processing.
- View and download processed images directly.

---

## Technologies Used

### Backend

- **Flask**: Web framework for handling image uploads and processing.
- **OpenCV**: Image processing library for various transformations.
- **scikit-image**: Used for superpixel segmentation.
- **Matplotlib**: For saving processed images.
- **Flask-CORS**: Handles cross-origin requests.

### Frontend

- **React**: For an interactive user interface.
- **Axios**: For making API requests to the Flask backend.

---

## Installation and Setup

### Prerequisites

- Python 3.8 or later
- Node.js and npm
- Virtual Environment (optional but recommended)

### Backend Setup

1. Navigate to the Backend Directory

```bash
cd Backend
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

4. Run the Flask server:

```bash
python app.py
```

### Frontend Setup

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
   npm install
```

3. Start the React development server:

```bash
   npm start
```
