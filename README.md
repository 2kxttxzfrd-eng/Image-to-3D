# Image to Lithophane 3D Converter

This is a Streamlit application that converts 2D images into 3D printable STL files (Lithophanes).

## Features
- Upload images (JPG, PNG)
- Convert to Lithophane (inverted heightmap) or standard Height Map
- Adjust dimensions (width, min/max thickness)
- Preview 3D mesh generation (internal processing)
- Download generated STL file

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```
or use the VS Code Task "Run Streamlit App".

## How it works
The application converts pixel brightness values into thickness variations. Darker pixels become thicker regions (blocking more light) for lithophanes, creating the image when back-lit.
