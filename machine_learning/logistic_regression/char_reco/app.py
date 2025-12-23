import streamlit as st
import numpy as np
import pickle
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# Load the model and scaler
@st.cache_resource
def load_model():
    with open('digit_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Title and description
st.title("🔢 Handwritten Digit Recognition")
st.markdown("Draw a digit (0-9) in the canvas below and click **Predict** to see the result!")

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Draw Here:")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",  # White background
        stroke_width=15,
        stroke_color="rgb(0, 0, 0)",  # Black stroke
        background_color="rgb(255, 255, 255)",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.markdown("### Prediction:")
    
    # Predict button
    if st.button("🎯 Predict Digit", type="primary", use_container_width=True):
        if canvas_result.image_data is not None:
            # Get the drawn image
            img_data = canvas_result.image_data
            
            # Convert to grayscale
            if len(img_data.shape) == 3:
                gray_image = cv2.cvtColor(img_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
            else:
                gray_image = img_data
            
            # Check if canvas is empty (all white)
            if np.mean(gray_image) > 250:
                st.warning("⚠️ Please draw a digit first!")
            else:
                # Invert colors (white on black like training data)
                gray_image = 255 - gray_image
                
                # Resize to 8x8 like the training data
                resized_image = cv2.resize(gray_image, (8, 8), interpolation=cv2.INTER_AREA)
                
                # Normalize to 0-16 range like the original digits dataset
                normalized_image = (resized_image / 255.0 * 16).astype(np.float64)
                
                # Flatten to 1D array
                flattened_image = normalized_image.flatten().reshape(1, -1)
                
                # Scale using the same scaler used for training
                scaled_image = scaler.transform(flattened_image)
                
                # Make prediction
                prediction = model.predict(scaled_image)[0]
                probabilities = model.predict_proba(scaled_image)[0]
                confidence = np.max(probabilities) * 100
                
                # Display results
                st.markdown(f"## Predicted Digit: **:blue[{prediction}]**")
                st.markdown(f"**Confidence:** {confidence:.1f}%")
                
                # Show probability distribution
                st.markdown("### Probability Distribution:")
                
                # Create a bar chart for probabilities
                prob_data = {f"Digit {i}": prob for i, prob in enumerate(probabilities)}
                
                # Filter and show only significant probabilities
                significant_probs = {k: v for k, v in prob_data.items() if v > 0.01}
                
                if significant_probs:
                    for digit_label, prob in significant_probs.items():
                        st.progress(float(prob), text=f"{digit_label}: {prob:.3f}")
                
                # Show the processed 8x8 image
                with st.expander("🔍 Show processed image (8x8)"):
                    st.image(resized_image, caption="8x8 processed image", width=200)
        else:
            st.warning("⚠️ Please draw a digit first!")
    
    # Clear button instruction
    st.info("💡 **Tip:** Use the 🗑️ icon at the top of the canvas to clear and draw again!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit | Logistic Regression Model</p>
        <p>Model Accuracy: 97.2%</p>
    </div>
""", unsafe_allow_html=True)
