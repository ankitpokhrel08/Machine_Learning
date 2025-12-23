import streamlit as st
import numpy as np
import pickle
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# Load the model and scaler
def load_model():
    # Get the directory where app.py is located
    current_dir = Path(__file__).parent
    model_path = current_dir / 'digit_model.pkl'
    scaler_path = current_dir / 'scaler.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load a sample from sklearn digits to verify preprocessing
    from sklearn.datasets import load_digits
    digits = load_digits()
    sample_image = digits.data[0].reshape(1, -1)  # First digit (should be 0)
    sample_scaled = scaler.transform(sample_image)
    sample_pred = model.predict(sample_scaled)[0]
    
    print(f"✓ Model loaded - Test prediction on sample digit 0: {sample_pred}")
    print(f"  Sample raw mean: {sample_image.mean():.2f}, Sample scaled mean: {sample_scaled.mean():.2f}")
    
    return model, scaler

model, scaler = load_model()

# Load sklearn digits for comparison
from sklearn.datasets import load_digits
digits_data = load_digits()

# Title and description
st.title("Handwritten Digit Recognition")
st.markdown("Draw a digit (0-9) in the canvas below and click **Predict** to see the result!")

# Add a test section
with st.expander("Model Test - Verify model works correctly"):
    test_idx = st.slider("Select a digit from sklearn dataset to test", 0, len(digits_data.images)-1, 0)
    test_image = digits_data.data[test_idx].reshape(1, -1)
    test_scaled = scaler.transform(test_image)
    test_pred = model.predict(test_scaled)[0]
    test_actual = digits_data.target[test_idx]
    
    col_test1, col_test2 = st.columns(2)
    with col_test1:
        # Normalize to 0-1 range for display
        test_img_normalized = digits_data.images[test_idx] / 16.0
        st.image(test_img_normalized, caption=f"Sklearn digit #{test_idx}", width=150)
    with col_test2:
        st.metric("Actual Label", test_actual)
        st.metric("Model Prediction", test_pred)
        if test_pred == test_actual:
            st.success("✅ Model works correctly!")
        else:
            st.error("❌ Model prediction is wrong!")


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
                # Invert colors (black on white -> white on black like training data)
                inverted_image = 255 - gray_image
                
                # Resize to 8x8 like the training data (sklearn digits dataset)
                resized_image = cv2.resize(inverted_image, (8, 8), interpolation=cv2.INTER_AREA)
                
                # Normalize to 0-16 range like the original sklearn digits dataset
                normalized_image = (resized_image / 255.0 * 16).astype(np.float64)
                
                # Flatten to 1D array (64 features)
                flattened_image = normalized_image.flatten().reshape(1, -1)
                
                # Scale using the same scaler used for training
                scaled_image = scaler.transform(flattened_image)
                
                # Make prediction
                prediction = model.predict(scaled_image)[0]
                probabilities = model.predict_proba(scaled_image)[0]
                confidence = np.max(probabilities) * 100
                
                # Display results with large font
                st.markdown(f"<h1 style='text-align: center; color: #1f77b4; font-size: 5em; margin: 0;'>{prediction}</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 1.2em;'><strong>Confidence:</strong> {confidence:.1f}%</p>", unsafe_allow_html=True)
                
                # Show probability distribution
                st.markdown("### Top Predictions:")
                
                # Get top 3 predictions
                top_3_indices = np.argsort(probabilities)[-3:][::-1]
                for idx in top_3_indices:
                    if probabilities[idx] > 0.01:
                        st.progress(float(probabilities[idx]), text=f"Digit {idx}: {probabilities[idx]:.3f}")
                
                # Show the processed 8x8 image
                with st.expander("🔍 Compare with Dataset"):
                    st.markdown("### Your Drawing vs Original Dataset")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Your Drawing:**")
                        # Use normalized image (0-16 range) divided by 16 for display
                        st.image(normalized_image / 16.0, caption="Your 8x8 digit", width=200)
                    
                    # Show a similar sklearn digit for comparison
                    with col_b:
                        st.markdown(f"**Dataset Digit {prediction}:**")
                        # Find first occurrence of the predicted digit in sklearn dataset
                        idx = np.where(digits_data.target == prediction)[0][0]
                        # Normalize to 0-1 range for display
                        sklearn_img_normalized = digits_data.images[idx] / 16.0
                        st.image(sklearn_img_normalized, caption=f"Original digit {prediction}", width=200)
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
