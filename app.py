import streamlit as st
import numpy as np
from PIL import Image
from stl import mesh
import tempfile
import os
import plotly.graph_objects as go

def image_to_lithophane(image, width_mm, thickness_mm, min_thickness_mm, contrast, inverted=True):
    # Convert image to grayscale
    img = image.convert('L')
    
    # Calculate dimensions
    aspect_ratio = img.height / img.width
    height_mm = width_mm * aspect_ratio
    
    # Downscale image if it's too large to keep processing reasonable
    # A resolution of 0.1mm per pixel is usually sufficient for 3D printing
    pixels_per_mm = 5 # 5 pixels per mm is high quality
    target_width_px = int(width_mm * pixels_per_mm)
    target_height_px = int(height_mm * pixels_per_mm)
    img = img.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)
    
    # Get pixel data as numpy array
    img_array = np.array(img)
    
    # Normalize pixel values
    if inverted:
        # Darker pixels = thicker (for light to pass through less)
        # 0 (black) -> thickness_mm
        # 255 (white) -> min_thickness_mm
        z_values = (1 - (img_array / 255.0)) * (thickness_mm - min_thickness_mm) + min_thickness_mm
    else:
        # Brighter pixels = thicker (standard bump map)
        z_values = (img_array / 255.0) * (thickness_mm - min_thickness_mm) + min_thickness_mm

    # Create vertices
    # X and Y coordinates range from 0 to width_mm/height_mm
    # The image array structure is (height, width), so rows are y and cols are x
    height_px, width_px = z_values.shape
    
    # Create grid of x, y coordinates
    x = np.linspace(0, width_mm, width_px)
    y = np.linspace(0, height_mm, height_px)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Vertices array: shape (height_px, width_px, 3)
    vertices = np.zeros((height_px, width_px, 3))
    vertices[:, :, 0] = x_grid
    vertices[:, :, 1] = np.flip(y_grid, axis=0) # Flip Y so image isn't upside down in 3D space
    vertices[:, :, 2] = z_values

    # Generate faces
    # Each quad (4 pixels) is made of 2 triangles
    # Number of quads is (height-1) * (width-1)
    # Number of triangles is 2 * (height-1) * (width-1)
    
    num_faces = 2 * (height_px - 1) * (width_px - 1)
    faces = np.zeros((num_faces, 3), dtype=int)
    
    # Calculate vertex indices
    # Vertex at (row, col) has index: row * width_px + col
    
    # We can vectorize the index calculation
    rows, cols = np.meshgrid(np.arange(height_px - 1), np.arange(width_px - 1), indexing='ij')
    
    # Top-left vertices for each quad
    v1 = rows * width_px + cols
    # Top-right
    v2 = v1 + 1
    # Bottom-left
    v3 = v1 + width_px
    # Bottom-right
    v4 = v3 + 1
    
    # Calculate total number of faces for valid solid volume
    # 1. Top surface: 2 * (height_px - 1) * (width_px - 1)
    # 2. Bottom surface: 2 (represented as 2 large triangles for flat back)
    # 3. 4 Side walls:
    #    Top edge: 2 * (width_px - 1)
    #    Bottom edge: 2 * (width_px - 1)
    #    Left edge: 2 * (height_px - 1)
    #    Right edge: 2 * (height_px - 1)
    
    num_top_faces = 2 * (height_px - 1) * (width_px - 1)
    num_side_faces = 2 * (2 * (width_px - 1) + 2 * (height_px - 1))
    num_bottom_faces = 2
    total_faces = num_top_faces + num_side_faces + num_bottom_faces
    
    # Create the mesh object
    lithophane_mesh = mesh.Mesh(np.zeros(total_faces, dtype=mesh.Mesh.dtype))
    
    # Flatten vertices for easy indexing
    vertices_flat = vertices.reshape(-1, 3)
    
    # --- 1. Top Surface ---
    
    # Flattening indices for fast construction
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()
    v3_flat = v3.flatten()
    v4_flat = v4.flatten()
    
    # Set vectors for Triangle 1 (v1, v2, v3)
    lithophane_mesh.vectors[:num_top_faces//2, 0] = vertices_flat[v1_flat]
    lithophane_mesh.vectors[:num_top_faces//2, 1] = vertices_flat[v2_flat]
    lithophane_mesh.vectors[:num_top_faces//2, 2] = vertices_flat[v3_flat]
    
    # Set vectors for Triangle 2 (v2, v4, v3)
    lithophane_mesh.vectors[num_top_faces//2:num_top_faces, 0] = vertices_flat[v2_flat]
    lithophane_mesh.vectors[num_top_faces//2:num_top_faces, 1] = vertices_flat[v4_flat]
    lithophane_mesh.vectors[num_top_faces//2:num_top_faces, 2] = vertices_flat[v3_flat]

    # --- 2. Side Walls & Bottom ---
    current_face_idx = num_top_faces
    
    # Helper to add a quad (2 triangles)
    def add_quad(p1, p2, p3, p4):
        nonlocal current_face_idx
        # Triangle 1: p1, p2, p3
        lithophane_mesh.vectors[current_face_idx, 0] = p1
        lithophane_mesh.vectors[current_face_idx, 1] = p2
        lithophane_mesh.vectors[current_face_idx, 2] = p3
        current_face_idx += 1
        
        # Triangle 2: p2, p4, p3
        lithophane_mesh.vectors[current_face_idx, 0] = p2
        lithophane_mesh.vectors[current_face_idx, 1] = p4
        lithophane_mesh.vectors[current_face_idx, 2] = p3
        current_face_idx += 1

    # Top edge (row 0)
    # Connect top vertices to bottom (z=0)
    for c in range(width_px - 1):
        t1 = vertices[0, c]
        t2 = vertices[0, c+1]
        b1 = t1.copy(); b1[2] = 0
        b2 = t2.copy(); b2[2] = 0
        # Winding order for outside normal
        add_quad(t1, t2, b1, b2) 

    # Bottom edge (row height-1)
    for c in range(width_px - 1):
        t1 = vertices[height_px-1, c+1] # traverse other way? No, just consistent indexing
        t2 = vertices[height_px-1, c]
        b1 = t1.copy(); b1[2] = 0
        b2 = t2.copy(); b2[2] = 0
        add_quad(t1, t2, b1, b2)

    # Left edge (col 0)
    for r in range(height_px - 1):
        t1 = vertices[r+1, 0]
        t2 = vertices[r, 0]
        b1 = t1.copy(); b1[2] = 0
        b2 = t2.copy(); b2[2] = 0
        add_quad(t1, t2, b1, b2)

    # Right edge (col width-1)
    for r in range(height_px - 1):
        t1 = vertices[r, width_px-1]
        t2 = vertices[r+1, width_px-1]
        b1 = t1.copy(); b1[2] = 0
        b2 = t2.copy(); b2[2] = 0
        add_quad(t1, t2, b1, b2)
        
    # --- 3. Bottom Face ---
    # Two large triangles to close the back
    # Corners
    c_tl = vertices[0, 0].copy(); c_tl[2] = 0
    c_tr = vertices[0, width_px-1].copy(); c_tr[2] = 0
    c_bl = vertices[height_px-1, 0].copy(); c_bl[2] = 0
    c_br = vertices[height_px-1, width_px-1].copy(); c_br[2] = 0
    
    # Facing down
    add_quad(c_bl, c_br, c_tl, c_tr)

    # Calculate normals for the mesh to ensure correct shading in 3D viewers
    # This is critical for the object to look "3D" and not just flat
    lithophane_mesh.update_normals()

    return lithophane_mesh, vertices

def main():
    st.set_page_config(page_title="Image to Lithophane Converter", page_icon="🖨️", layout="wide")
    
    st.title("🖨️ Image to Lithophane 3D Converter")
    st.markdown("""
    Convert your favorite photos into 3D printable lithophanes.
    *Upload an image, adjust settings, and see a predicted 3D representation.*
    """)
    
    # Initialize session state for storing the generated mesh
    if 'generated_file_path' not in st.session_state:
        st.session_state.generated_file_path = None
    if 'preview_data' not in st.session_state:
        st.session_state.preview_data = None
    if 'mesh_bounds' not in st.session_state:
         st.session_state.mesh_bounds = None
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            st.divider()
            st.subheader("Settings")
            width_mm = st.sidebar.slider("Width (mm)", 50, 200, 100)
            
            with st.expander("Advanced Dimensions", expanded=True):
                 width_mm_val = st.slider("Model Width (mm)", 50, 200, 100, key="width_slider")
                 max_thickness = st.slider("Max Thickness (mm)", 1.0, 10.0, 3.0, 0.1)
                 min_thickness = st.slider("Min Thickness (mm)", 0.4, 4.0, 0.8, 0.1)
            
            mode = st.radio("Mode", ["Lithophane (Inverted)", "Height Map (Standard)"])
            inverted = mode == "Lithophane (Inverted)"
            
            generate_btn = st.button("Generate 3D Model", type="primary")

    with col2:
        if uploaded_file is not None and generate_btn:
            with st.spinner("Processing image and generating 3D geometry..."):
                try:
                    litho_mesh, vertices = image_to_lithophane(
                        image, 
                        width_mm_val, 
                        max_thickness, 
                        min_thickness, 
                        1.0, 
                        inverted
                    )
                    
                    # Save STL
                    temp_dir = tempfile.gettempdir()
                    temp_path = os.path.join(temp_dir, f"lithophane_{os.urandom(4).hex()}.stl")
                    litho_mesh.save(temp_path)
                    st.session_state.generated_file_path = temp_path
                    
                    # Prepare Visualization Data
                    # Downsample for performance
                    ds = 4 # Downsample factor (lower = higher detailed preview, slower)
                    z_data = vertices[:, :, 2]
                    
                    # Store data for plotly
                    x_data = vertices[:, :, 0]
                    y_data = vertices[:, :, 1]
                    
                    st.session_state.preview_data = {
                        'x': x_data[::ds, ::ds],
                        'y': y_data[::ds, ::ds],
                        'z': z_data[::ds, ::ds]
                    }
                    st.session_state.mesh_bounds = {
                        'width': width_mm_val,
                        'height': width_mm_val * (image.height / image.width)
                    }
                    
                    st.success("Analysis Complete!")
                    
                except Exception as e:
                    st.error(f"Error: {e}")

        # Display Visualization Area
        if st.session_state.generated_file_path:
             st.subheader("3D Surface Prediction")
             
             if st.session_state.preview_data:
                data = st.session_state.preview_data
                
                # Create interactive 3D surface plot
                fig = go.Figure(data=[go.Surface(
                    z=data['z'], 
                    x=data['x'], 
                    y=data['y'],
                    colorscale='Viridis' if not inverted else 'Gray',
                    reversescale=inverted,
                    showscale=False
                )])

                fig.update_layout(
                    title='Predicted 3D Geometry Surface',
                    autosize=True,
                    width=800,
                    height=600,
                    scene = dict(
                        xaxis_title='Width (mm)',
                        yaxis_title='Height (mm)',
                        zaxis_title='Thickness (mm)',
                        aspectmode='data'
                    ),
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 Rotate, zoom, and pan the chart above to inspect the texture details.")

             # Download Button
             with open(st.session_state.generated_file_path, "rb") as f:
                btn = st.download_button(
                    label="📥 Download STL File",
                    data=f,
                    file_name="lithophane.stl",
                    mime="model/stl"
                )

if __name__ == "__main__":
    main()
