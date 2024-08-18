# Rectangle and Intensity Detection Project

This project is designed to detect rectangles in images and evaluate their intensity contrasts using a variety of computer vision techniques. It integrates edge detection, Hough Line Transform, corner detection, and intensity checks using the Structural Similarity Index (SSIM) along with other advanced methods.

## Features

- **Edge Detection**: Utilizes the Canny edge detection algorithm to highlight edges within the image.
- **Hough Line Transform**: Detects straight lines, enhancing the accuracy of rectangle detection.
- **Corner Detection**: Implements Harris corner detection to identify potential rectangle corners.
- **Rectangle Detection**: Combines contour approximation with Hough Line Transform for precise rectangle detection.
- **Intensity Check**: Uses SSIM, histogram comparison, and gradient magnitude analysis to ensure the significance of detected rectangles.

## Installation

### First-Time Setup:

1. **Create a virtual environment**:
   ```bash
   python -m venv myenv
   ```

2. **Activate the virtual environment**:
   - On **Windows**:
     ```bash
     myenv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source myenv/bin/activate
     ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the main script**:
   ```bash
   python main.py
   ```

### If the Virtual Environment is Already Created:

1. **Activate the virtual environment**:
   - On **Windows**:
     ```bash
     myenv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source myenv/bin/activate
     ```

2. **Run the main script**:
   ```bash
   python main.py
   ```

## Dependencies

This project relies on the following Python packages, which will be installed when you run `pip install -r requirements.txt`:

- `opencv-python`: For image processing.
- `numpy`: For numerical operations and array manipulation.
- `matplotlib`: For visualizing images and plotting results.
- `scikit-image`: For advanced image processing, including SSIM.

## Project Structure

```
├── main.py                # Main script containing the rectangle detection and intensity check logic
├── README.md              # Project documentation
├── requirements.txt       # List of dependencies
└── images/                # Directory for storing test images
```

## Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

### Explanation of Changes:
- **Installation Instructions**: I've clarified the instructions for first-time setup and running the script when the virtual environment is already created.
- **Project Structure**: Updated the project structure to reflect the typical files you would have.
- **General Refinements**: Improved wording and structure for clarity and conciseness.

You can copy this refined `README.md` into your project. If you have any further questions or need additional sections, feel free to ask!
