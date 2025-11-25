# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Rice Disease Detector",
    page_icon="🌾",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('rice_disease_cnn.h5')
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction - handles RGBA images"""
    # Convert RGBA (4 channels) to RGB (3 channels) if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Resize to match model input
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Ensure we have exactly 3 channels
    if img_array.shape[-1] != 3:
        img_array = img_array[:, :, :3]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def main():
    st.title("Rice Leaf Disease Detector")
    st.markdown("Upload an image of a rice leaf for AI disease analysis")
    
    # Load model
    model = load_model()
    class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
    
    if model is None:
        st.error("Model not found! Make sure 'rice_disease_cnn.h5' is in your project folder.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a rice leaf image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
        
        with col2:
            with st.spinner('Analyzing image...'):
                try:
                    # Preprocess and predict
                    processed_image = preprocess_image(image)
                    predictions = model.predict(processed_image, verbose=0)
                    predicted_class = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class]
                    
                    # DISPLAY FINAL RESULT SEPARATELY
                    st.subheader("FINAL DIAGNOSIS")
                    
                    # Display based on disease type
                    if class_names[predicted_class] == 'Bacterial leaf blight':
                        st.error(f"DISEASE DETECTED: BACTERIAL LEAF BLIGHT")
                    elif class_names[predicted_class] == 'Brown spot':
                        st.warning(f"DISEASE DETECTED: BROWN SPOT")
                    elif class_names[predicted_class] == 'Leaf smut':
                        st.warning(f"DISEASE DETECTED: LEAF SMUT")
                    else:
                        st.success(f"HEALTHY LEAF")
                    
                    # Confidence display
                    st.metric(
                        label="AI Confidence Level",
                        value=f"{confidence:.2%}",
                        delta="High Confidence" if confidence > 0.8 else "Moderate Confidence"
                    )
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        
        # DETAILED ANALYSIS SECTION (Separate)
        st.markdown("---")
        st.subheader("Detailed Analysis")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.write("Confidence Breakdown:")
            for i, (class_name, conf) in enumerate(zip(class_names, predictions[0])):
                # Highlight the predicted class
                if i == predicted_class:
                    st.success(f"PREDICTED: {class_name}: {conf:.2%}")
                else:
                    st.write(f"{class_name}: {conf:.2%}")
        
        with col4:
            st.write("Recommendation:")
            disease = class_names[predicted_class]
            if disease == 'Bacterial leaf blight':
                st.info("""
                Treatment Advice:
                - Use copper-based bactericides
                - Remove infected plants
                - Practice field sanitation
                - Avoid overhead irrigation
                """)
            elif disease == 'Brown spot':
                st.info("""
                Treatment Advice:
                - Apply fungicides
                - Ensure proper nutrient management
                - Improve soil drainage
                - Use resistant varieties
                """)
            elif disease == 'Leaf smut':
                st.info("""
                Treatment Advice:
                - Use resistant varieties
                - Practice crop rotation
                - Remove infected debris
                - Apply appropriate fungicides
                """)
            else:
                st.success("""
                Plant is Healthy!
                - Continue good farming practices
                - Regular monitoring recommended
                - Maintain proper nutrition
                """)

if __name__ == "__main__":
    main()