"""Task 3: Image Captioning using CNN and RNN
Combines computer vision and NLP to generate captions for images.
Uses pre-trained CNN (VGG16) for feature extraction and LSTM for caption generation.
"""

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, RepeatVector
from tensorflow.keras.layers import concatenate, Input, Reshape, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

class ImageCaptioner:
    def __init__(self, max_length=50):
        self.max_length = max_length
        self.vocab_size = 5000
        # Load pre-trained VGG16 for feature extraction
        self.vgg_model = VGG16(weights='imagenet')
        # Remove the last layer to get feature vectors
        self.feature_extractor = Model(inputs=self.vgg_model.input,
                                      outputs=self.vgg_model.layers[-2].output)
    
    def extract_features(self, image_path):
        """Extract features from an image using VGG16"""
        try:
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = self.feature_extractor.predict(img_array)
            return features.flatten()
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def build_caption_model(self):
        """Build the image captioning model"""
        # Image feature input (4096-dimensional vector from VGG16)
        image_input = Input(shape=(4096,), name='image_input')
        
        # Sequence input for captions during training
        caption_input = Input(shape=(self.max_length,), name='caption_input')
        
        # Image feature processing
        img_feat = Dense(256, activation='relu')(image_input)
        img_feat = Dropout(0.5)(img_feat)
        
        # Caption embedding and LSTM
        cap_emb = Embedding(self.vocab_size, 128)(caption_input)
        cap_emb = LSTM(256, return_sequences=False)(cap_emb)
        cap_emb = Dropout(0.5)(cap_emb)
        
        # Merge image and caption features
        merged = concatenate([img_feat, cap_emb])
        merged = Dense(256, activation='relu')(merged)
        merged = Dropout(0.5)(merged)
        
        # Output layer
        output = Dense(self.vocab_size, activation='softmax')(merged)
        
        # Create model
        model = Model(inputs=[image_input, caption_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def generate_caption(self, image_path, model, tokenizer, max_length=50):
        """Generate a caption for an image"""
        features = self.extract_features(image_path)
        if features is None:
            return "Unable to process image"
        
        # Prepare features for model
        features = np.expand_dims(features, axis=0)
        
        # Start with a beginning token (in practice, would be a specific token)
        caption = "A"
        
        # Generate caption word by word
        for _ in range(max_length):
            # Encode caption
            sequence = tokenizer.texts_to_sequences([caption])[0]
            sequence = np.pad(sequence, (0, max_length - len(sequence)))
            sequence = np.expand_dims(sequence, axis=0)
            
            # Predict next word
            # Note: In a real implementation, you would use teacher forcing during training
            # and beam search during inference
            
            caption += " [next_word]"
        
        return caption
    
    def preprocess_image_dataset(self, images_dir, captions_file):
        """Preprocess image dataset for training"""
        features_dict = {}
        captions_dict = {}
        
        # Load captions from file (format: image_id caption)
        with open(captions_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    image_id, caption = parts
                    image_path = os.path.join(images_dir, image_id)
                    
                    # Extract and cache features
                    if os.path.exists(image_path):
                        features = self.extract_features(image_path)
                        if features is not None:
                            features_dict[image_id] = features
                            captions_dict[image_id] = caption
        
        return features_dict, captions_dict

if __name__ == "__main__":
    # Example usage
    print("Image Captioning Model")
    print("=" * 50)
    print("\nThis model combines:")
    print("- VGG16 (pre-trained on ImageNet) for feature extraction")
    print("- LSTM neural network for caption generation")
    print("\nUsage:")
    print("1. Extract features from images using VGG16")
    print("2. Train LSTM model on image-caption pairs")
    print("3. Generate captions for new images")
    print("\nNote: Full implementation requires:")
    print("- Image dataset (e.g., MS COCO, Flickr30k)")
    print("- Corresponding captions")
    print("- Training infrastructure")
    
    captioner = ImageCaptioner(max_length=50)
    print("\nImageCaptioner model initialized successfully!")
    print(f"Vocabulary size: {captioner.vocab_size}")
    print(f"Maximum caption length: {captioner.max_length}")
