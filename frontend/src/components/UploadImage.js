import React, { useState } from "react";
import axios from "axios";
import "./ImageUpload.css";

const ImageUpload = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [std1, setStd1] = useState(1);
  const [std2, setStd2] = useState(2);
  const [nSuperpixels, setNSuperpixels] = useState(200);
  const [imageResults, setImageResults] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const formatTitle = (key) => {
    return key
      .replace(/([A-Z])/g, " $1")
      .replace(/^./, (str) => str.toUpperCase());
  };

  const handleImageChange = (e) => {
    setSelectedImage(e.target.files[0]);
    setImageResults({});
    setError("");
  };

  const handleImageUpload = async () => {
    if (!selectedImage) {
      setError("Please select an image before uploading.");
      return;
    }

    if (std2 <= std1) {
      setError("std2 must be greater than std1.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedImage);
    formData.append("std1", std1);
    formData.append("std2", std2);
    formData.append("nSuperpixels", nSuperpixels);

    setLoading(true);
    setError("");

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setImageResults({
        rgb: `http://127.0.0.1:5000${
          response.data.rgb_image_url
        }?t=${Date.now()}`,
        gray: `http://127.0.0.1:5000${
          response.data.gray_image_url
        }?t=${Date.now()}`,
        canny: `http://127.0.0.1:5000${
          response.data.canny_image_url
        }?t=${Date.now()}`,
        sparse: `http://127.0.0.1:5000${
          response.data.sparse_image_url
        }?t=${Date.now()}`,
        mag1: `http://127.0.0.1:5000${
          response.data.mag1_image_url
        }?t=${Date.now()}`,
        mag2: `http://127.0.0.1:5000${
          response.data.mag2_image_url
        }?t=${Date.now()}`,
        slic: `http://127.0.0.1:5000${
          response.data.slic_image_url
        }?t=${Date.now()}`,
        denseSlic: `http://127.0.0.1:5000${
          response.data.dense_slic_image_url
        }?t=${Date.now()}`,
        snic: `http://127.0.0.1:5000${
          response.data.snic_image_url
        }?t=${Date.now()}`,
        denseSnic: `http://127.0.0.1:5000${
          response.data.dense_snic_image_url
        }?t=${Date.now()}`,
      });
    } catch (err) {
      setError("Error uploading the image. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="image-upload">
      <h2>Upload Image for Processing</h2>
      <input type="file" onChange={handleImageChange} />
      <div>
        <label style={{ marginRight: "20px" }}>
          Std1:
          <input
            type="number"
            value={std1}
            onChange={(e) => setStd1(parseFloat(e.target.value))}
            min="0"
            step="0.1"
          />
        </label>
        <label style={{ marginRight: "20px" }}>
          Std2:
          <input
            type="number"
            value={std2}
            onChange={(e) => setStd2(parseFloat(e.target.value))}
            min={std1 + 0.1}
            step="0.1"
          />
        </label>
        <label>
          Number of Superpixels:
          <input
            type="number"
            value={nSuperpixels}
            onChange={(e) => setNSuperpixels(parseInt(e.target.value))}
            min="10"
          />
        </label>
      </div>
      <button
        onClick={handleImageUpload}
        disabled={loading}
        style={{ marginTop: "20px" }}
      >
        {loading ? "Processing..." : "Upload"}
      </button>

      {error && <p className="error">{error}</p>}
      {loading && <div className="spinner"></div>}

      {Object.keys(imageResults).length > 0 && (
        <div className="results">
          {Object.entries(imageResults).map(([key, url]) => (
            <div key={key} className="image-container">
              <h3>{formatTitle(key)} Image</h3>
              <img src={url} alt={`${key} Image`} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
