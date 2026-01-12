import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
from stl import mesh
import tempfile
import os
import plotly.graph_objects as go

def create_circular_mask(h, w):
    center = (int(w / 2), int(h / 2))
    radius = min(center[0], center[1], w - center[0], h - center[1])
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    
    mask = dist_from_center <= radius
    return mask

def add_text_to_image(image, text, size=50, position='Bottom', color_val=0):
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Try to load a font, fallback to default if necessary
    try:
        # Try loading a common font
        # For macOS/Linux, paths might vary. Let's try a few standard ones.
        font_paths = [
            "Arial.ttf", 
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf" # Windows
        ]
        
        font = None
        for path in font_paths:
            try:
                if os.path.exists(path) or (not os.path.isabs(path)):
                    font = ImageFont.truetype(path, size)
                    break
            except:
                continue
                
        if font is None:
             font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Calculate text size
    if hasattr(draw, 'textbbox'):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top
    else:
        # Fallback for older PIL
        text_w, text_h = draw.textsize(text, font=font)
    
    W, H = img_draw.size
    
    x, y = 0, 0
    padding = int(min(W, H) * 0.05)
    
    if position == 'Top':
        x = (W - text_w) / 2
        y = padding
    elif position == 'Bottom':
        x = (W - text_w) / 2
        y = H - text_h - padding
    elif position == 'Center':
        x = (W - text_w) / 2
        y = (H - text_h) / 2
        
    # Determine fill color based on image mode
    fill_color = color_val
    if img_draw.mode == 'RGB':
        fill_color = (color_val, color_val, color_val)
    elif img_draw.mode == 'RGBA':
        fill_color = (color_val, color_val, color_val, 255)
        
    draw.text((x, y), text, font=font, fill=fill_color)
    
    return img_draw

def image_to_lithophane(image, width_mm, thickness_mm, min_thickness_mm, contrast, inverted=True, shape='Rectangle'):
    # Convert image to grayscale
    img = image.convert('L')
    
    # Calculate dimensions
    if shape == 'Round':
         # For round images, we make it a square first for processing
         # Crop to center square
         min_side = min(img.width, img.height)
         left = (img.width - min_side)/2
         top = (img.height - min_side)/2
         right = (img.width + min_side)/2
         bottom = (img.height + min_side)/2
         img = img.crop((left, top, right, bottom))
         aspect_ratio = 1.0
    else:
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

    # Reshape vertices to list for indexing
    vertices_flat = vertices.reshape((-1, 3))

    # Calculate vertex indices
    # Vertex at (row, col) has index: row * width_px + col
    # We can vectorize the index calculation
    rows, cols = np.meshgrid(np.arange(height_px - 1), np.arange(width_px - 1), indexing='ij')
    
    # Masking for shapes
    valid_quads_mask = np.ones(rows.shape, dtype=bool)

    if shape == 'Round':
         # Calculate radius in index space (approximate)
         ci_r = (height_px - 1) / 2
         ci_c = (width_px - 1) / 2
         # Use slightly smaller radius to ensure walls are within bounds
         rad_i_sq = (min(height_px - 1, width_px - 1) / 2)**2
         
         dist_sq_quads = (rows - ci_r)**2 + (cols - ci_c)**2
         valid_quads_mask = dist_sq_quads <= rad_i_sq
         
         rows = rows[valid_quads_mask]
         cols = cols[valid_quads_mask]
    else:
         rows = rows.flatten()
         cols = cols.flatten()
    
    # Generate Top Surface Triangles
    v1 = rows * width_px + cols
    v2 = v1 + 1
    v3 = v1 + width_px
    v4 = v3 + 1
    
    curr_num_faces = len(v1) * 2
    
    t1_v1 = vertices_flat[v1]
    t1_v2 = vertices_flat[v2]
    t1_v3 = vertices_flat[v3]
    
    t2_v1 = vertices_flat[v2]
    t2_v2 = vertices_flat[v4]
    t2_v3 = vertices_flat[v3]
    
    all_triangles = []
    
    top_tris = np.zeros((curr_num_faces, 3, 3))
    # First triangle of quad: v1, v2, v3
    top_tris[0::2, 0] = t1_v1; top_tris[0::2, 1] = t1_v2; top_tris[0::2, 2] = t1_v3
    # Second triangle of quad: v2, v4, v3
    top_tris[1::2, 0] = t2_v1; top_tris[1::2, 1] = t2_v2; top_tris[1::2, 2] = t2_v3
    
    all_triangles.append(top_tris)
    
    walls = []
    
    if shape == 'Rectangle':
        # Add walls for Rectangle
        # Left/Right edges
        for r in range(height_px - 1):
             # Left edge (col 0)
             t1 = vertices[r+1, 0]; t2 = vertices[r, 0]
             b1 = t1.copy(); b1[2] = 0; b2 = t2.copy(); b2[2] = 0
             walls.append([t1, t2, b1]); walls.append([t2, b2, b1])
             
             # Right edge (col width-1)
             t1 = vertices[r, width_px-1]; t2 = vertices[r+1, width_px-1]
             b1 = t1.copy(); b1[2] = 0; b2 = t2.copy(); b2[2] = 0
             walls.append([t1, t2, b1]); walls.append([t2, b2, b1])
        
        # Top/Bottom edges
        for c in range(width_px - 1):
             # Top edge (row 0)
             t1 = vertices[0, c]; t2 = vertices[0, c+1]
             b1 = t1.copy(); b1[2] = 0; b2 = t2.copy(); b2[2] = 0
             walls.append([t1, t2, b1]); walls.append([t2, b2, b1])

             # Bottom edge (row height-1)
             t1 = vertices[height_px-1, c+1]; t2 = vertices[height_px-1, c]
             b1 = t1.copy(); b1[2] = 0; b2 = t2.copy(); b2[2] = 0
             walls.append([t1, t2, b1]); walls.append([t2, b2, b1])

    elif shape == 'Round':
         # Add walls for Round (Grid Boundary)
         pad_mask = np.pad(valid_quads_mask, pad_width=1, mode='constant', constant_values=False)
         
         def add_wall_quad(p1, p2):
             b1 = p1.copy(); b1[2] = 0
             b2 = p2.copy(); b2[2] = 0
             walls.append([p1, p2, b1])
             walls.append([p2, b2, b1])
             
         full_rows, full_cols = np.nonzero(valid_quads_mask)
         
         for i in range(len(full_rows)):
             r = full_rows[i]
             c = full_cols[i]
             
             # Check neighbors in the PADDED mask
             # Current quad is at [r+1, c+1] in pad_mask
             
             # Top Neighbor (r-1, c) -> pad_mask[r, c+1]
             if not pad_mask[r, c+1]:
                  p1 = vertices[r, c]; p2 = vertices[r, c+1]
                  add_wall_quad(p1, p2)
                  
             # Bottom Neighbor (r+1, c) -> pad_mask[r+2, c+1]
             if not pad_mask[r+2, c+1]:
                  p1 = vertices[r+1, c+1]; p2 = vertices[r+1, c]
                  add_wall_quad(p1, p2)
            
             # Left Neighbor (r, c-1) -> pad_mask[r+1, c]
             if not pad_mask[r+1, c]:
                  p1 = vertices[r+1, c]; p2 = vertices[r, c]
                  add_wall_quad(p1, p2)
                  
             # Right Neighbor (r, c+1) -> pad_mask[r+1, c+2]
             if not pad_mask[r+1, c+2]:
                  p1 = vertices[r, c+1]; p2 = vertices[r+1, c+1]
                  add_wall_quad(p1, p2)
         
         # Bottom Cap (Mirror of Top Surface, flattened to Z=0)
         bottom_tris = top_tris.copy()
         bottom_tris[:, :, 2] = 0
         # Reverse winding order for bottom faces to point down
         bottom_tris[:, [1, 2]] = bottom_tris[:, [2, 1]]
         all_triangles.append(bottom_tris)
         
    if shape == 'Rectangle':
         # Bottom Cap for Rectangle (Mirror of Top Surface, flattened to Z=0)
         # Using this instead of the single quad ensures vertices align perfectly (watertight)
         bottom_rect_tris = top_tris.copy()
         bottom_rect_tris[:, :, 2] = 0
         bottom_rect_tris[:, [1, 2]] = bottom_rect_tris[:, [2, 1]]
         all_triangles.append(bottom_rect_tris)

    if walls:
        all_triangles.append(np.array(walls))
    
    final_tris = np.concatenate(all_triangles)
    
    lithophane_mesh = mesh.Mesh(np.zeros(final_tris.shape[0], dtype=mesh.Mesh.dtype))
    lithophane_mesh.vectors = final_tris
    
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
            
            # Dimensions
            width_mm_val = st.slider("Model Width (mm)", 50, 200, 100, key="width_slider")
            
            with st.expander("Advanced Thickness", expanded=False):
                 max_thickness = st.slider("Max Thickness (mm)", 1.0, 10.0, 3.0, 0.1)
                 min_thickness = st.slider("Min Thickness (mm)", 0.4, 4.0, 0.8, 0.1)
            
            mode = st.radio("Mode", ["Lithophane (Inverted)", "Height Map (Standard)"])
            inverted = mode == "Lithophane (Inverted)"
            
            shape_option = st.radio("Shape", ["Rectangle", "Round"], horizontal=True)

            with st.expander("Add Custom Text", expanded=False):
                 add_text = st.checkbox("Enable Text Overlay")
                 if add_text:
                     custom_text = st.text_input("Text Content", "Happy Birthday!")
                     text_size = st.slider("Font Size", 10, 200, 50)
                     text_pos = st.selectbox("Position", ["Top", "Center", "Bottom"], index=2)
                     text_color = st.selectbox("Color", ["White", "Black"], index=0)
                     
                     # Map selection to pixel value
                     # For Lithophane (Inverted):
                     #   Darker (Black) -> Thicker -> Blocks light -> Dark on lithophane
                     #   Lighter (White) -> Thinner -> Lets light -> Bright on lithophane
                     
                     # BUT our logic is: 
                     # if inverted=True:
                     #    z = (1 - val/255) ...
                     #    0 (Black) -> 1 -> Max Thickness -> Blocks Light -> Dark to eye
                     #    255 (White) -> 0 -> Min Thickness -> Passes Light -> Bright to eye
                     
                     # So if user wants text to be "White" (Bright), we need pixel value 255
                     # If user wants text to be "Black" (Dark), we need pixel value 0
                     
                     if "White" in text_color:
                         text_val = 255
                     else:
                         text_val = 0
            
            generate_btn = st.button("Generate 3D Model", type="primary")

    with col2:
        if uploaded_file is not None and generate_btn:
            with st.spinner("Processing image and generating 3D geometry..."):
                try:
                    # Pre-process image with text if desired
                    processed_image = image
                    if add_text and custom_text:
                         processed_image = add_text_to_image(image, custom_text, text_size, text_pos, text_val)
                         # Show user the modified image
                         # st.image(processed_image, caption='Image with Text Overlay', width=200)

                    litho_mesh, vertices = image_to_lithophane(
                        processed_image, 
                        width_mm_val, 
                        max_thickness, 
                        min_thickness, 
                        1.0, 
                        inverted,
                        shape_option
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
                    
                    if add_text and custom_text:
                        st.info(f"✅ Generated with text: '{custom_text}'")
                    else:
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
