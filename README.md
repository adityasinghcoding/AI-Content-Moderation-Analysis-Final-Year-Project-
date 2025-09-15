# ACMA: AI Content Moderation Analysis (Final Year Project)

ACMA is an advanced AI-driven content moderation system designed to detect and analyze toxicity, inappropriate visuals, and violence across multiple content types, including text, images, audio, and video. This project leverages state-of-the-art deep learning and natural language processing techniques to provide a robust and extensible moderation toolkit for modern content platforms.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Model Download & Usage](#model-download--usage)
- [Setup Instructions](#setup-instructions)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Features

- **Multi-modal Moderation:** Analyze text, images, audio, and video for harmful or inappropriate content.
- **Toxicity & Violence Detection:** Detects toxic language, visual violence, and explicit imagery.
- **OCR Support:** Extracts and analyzes text from images using EasyOCR.
- **Modular Design:** Easily extend to support new moderation models or data types.

---

## Tech Stack

- **Languages:** Python
- **Libraries/Frameworks:** TensorFlow, Keras, OpenCV, EasyOCR, NLTK, Flask, scikit-learn

---

## Project Structure

```
/AI-Content-Moderation-Analysis-Final-Year-Project-
│
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── app/                     # Flask web server and API code
├── models/                  # Pre-trained model files and model-related code
├── data/                    # Sample data for testing (text, images, etc.)
├── utils/                   # Utility scripts (preprocessing, helpers)
├── notebooks/               # Jupyter notebooks for exploration and prototyping
├── static/                  # Static files for web interface (if any)
└── ...                      # Other files and directories
```

---

## Model Download & Usage

To use or download pre-trained models for this project:

- **Download Link:** [Google Drive Folder](https://drive.google.com/drive/u/0/folders/1CDdlclL76CC-JeheTjV6Bn1hgLfL7y6Z)

Place the downloaded model files in the appropriate `/models` directory as per the project structure.

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adityasinghcoding/AI-Content-Moderation-Analysis-Final-Year-Project-.git
   cd AI-Content-Moderation-Analysis-Final-Year-Project-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare models:**
   - Download models from the [provided Drive link](https://drive.google.com/drive/u/0/folders/1CDdlclL76CC-JeheTjV6Bn1hgLfL7y6Z).
   - Place them in the `/models` directory.

4. **Run the application:**
   ```bash
   python app/main.py
   ```
   _Or follow the instructions in the `/app` directory if using Flask or a different entry point._

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, new features, or bug fixes.

---

## Contact

- **Author:** Aditya Singh ([GitHub](https://github.com/adityasinghcoding))
- _For questions or support, please open an issue in the repository._
