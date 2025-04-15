import streamlit as st
from PIL import Image
import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage import convolve
import traceback

# Define Sobel filters
sx = np.array([
    [-0.125, -0.25, -0.125],
    [0.0, 0.0, 0.0],
    [0.125, 0.25, 0.125],
])

sy = np.array([ 
    [-0.125,  0.0,  0.125],
    [-0.25,   0.0,  0.25],
    [-0.125,  0.0,  0.125],
])

def least_edgy(e):
    n, m = e.shape
    dirs = np.zeros(e.shape, int)
    least_e = np.zeros(e.shape, np.float64)
    least_e[-1] = e[-1]
    
    for i in range(n - 2, -1, -1):
        for j in range(0, m):
            least_e[i][j] = min(
                least_e[i + 1][j], 
                least_e[i + 1][max(j - 1, 0)],
                least_e[i + 1][min(m - 1, j + 1)],
            ) + e[i][j] 
            dir = 0
            if j > 0 and least_e[i + 1][j - 1] < least_e[i + 1][j]:
                dir = -1
            if j < m - 1 and least_e[i + 1][j + dir] > least_e[i + 1][j + 1]:
                dir = 1
            dirs[i][j] = dir
    return least_e, dirs

def shorten(img_array, n):
    """Optimized seam carving function"""
    img = img_array.copy()
    
    # Convert to float32 for better performance
    img = img.astype(np.float32)
    
    # Handle RGBA images by converting to RGB
    if img.shape[2] == 4:
        img = img[:, :, :3]  # Keep only RGB channels
    
    # Pre-calculate gray image
    greyimg = rgb2gray(img)
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for i in range(n):
        # Update progress
        progress_bar.progress((i + 1) / n)
        progress_text.text(f'Processing seam {i + 1}/{n}')
        
        # Calculate energy map using vectorized operations
        dx = np.absolute(convolve(greyimg, sx))
        dy = np.absolute(convolve(greyimg, sy))
        energy_map = dx + dy
        
        # Find minimum seam
        least_e, dirs = least_edgy(energy_map)
        minId = np.argmin(least_e[0])
        
        # Create mask for removing seam
        mask = np.ones_like(img, dtype=bool)
        for row in range(img.shape[0]):
            mask[row, minId] = False
            minId += dirs[row][minId]
        
        # Remove seam using boolean indexing
        img = img[mask].reshape((img.shape[0], img.shape[1] - 1, img.shape[2]))
        greyimg = greyimg[mask[:,:,0]].reshape((greyimg.shape[0], greyimg.shape[1] - 1))
    
    progress_bar.empty()
    progress_text.empty()
    
    # Calculate final energy map for visualization
    dx = np.absolute(convolve(greyimg, sx))
    dy = np.absolute(convolve(greyimg, sy))
    final_energy = dx + dy
    
    return img.astype(np.uint8), final_energy

def main():
    st.title("Seam Carving Application")
    
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            original_size = image.size
            st.sidebar.write(f"Original size: {original_size[0]} x {original_size[1]}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            n_seams = st.slider("Number of seams to remove", 0, min(100, original_size[1]), 0)
            
            if n_seams > 0:
                img_array = np.array(image)
                processed_imgs = shorten(img_array, n_seams)
                
                with col2:
                    st.subheader("Processed Image")
                    carved_img = Image.fromarray(processed_imgs[0].astype('uint8'))
                    new_size = carved_img.size
                    st.write(f"New size: {new_size[0]} x {new_size[1]}")
                    st.write(f"Pixels removed: {original_size[0] * original_size[1] - new_size[0] * new_size[1]}")
                    
                    view_option = st.radio(
                        "Select View",
                        ["Carved Image", "Energy Map", "Difference"],
                        horizontal=True
                    )
                    
                    if view_option == "Carved Image":
                        st.image(carved_img, use_container_width=True)
                    elif view_option == "Energy Map":
                        energy_map = processed_imgs[1]
                        normalized_map = ((energy_map - energy_map.min()) * 255 / 
                                       (energy_map.max() - energy_map.min())).astype('uint8')
                        energy_img = Image.fromarray(normalized_map)
                        st.image(energy_img, use_container_width=True)
                    else:
                        # Show difference between original and carved
                        diff_img = Image.fromarray(
                            np.absolute(
                                np.array(image.resize(carved_img.size)) - 
                                np.array(carved_img)
                            ).astype('uint8')
                        )
                        st.image(diff_img, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.error(f"Stack trace:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
