import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# List of Food Classes
food_classes = [
    "Apple pie", "Baby back ribs", "Baklava", "Beef carpaccio", "Beef tartare",
    "Beet salad", "Beignets", "Bibimbap", "Bread pudding", "Breakfast burrito",
    "Bruschetta", "Caesar salad", "Cannoli", "Caprese salad", "Carrot cake", "Ceviche",
    "Cheesecake", "Cheese plate", "Chicken curry", "Chicken quesadilla", "Chicken wings",
    "Chocolate cake", "Chocolate mousse", "Churros", "Clam chowder", "Club sandwich",
    "Crab cakes", "Creme brulee", "Croque madame", "Cup cakes", "Deviled eggs",
    "Donuts", "Dumplings", "Edamame", "Eggs benedict", "Escargots", "Falafel",
    "Filet mignon", "Fish and chips", "Foie gras", "French fries", "French onion soup",
    "French toast", "Fried calamari", "Fried rice", "Frozen yogurt", "Garlic bread",
    "Gnocchi", "Greek salad", "Grilled cheese sandwich", "Grilled salmon", "Guacamole",
    "Gyoza", "Hamburger", "Hot and sour soup", "Hot dog", "Huevos rancheros", "Hummus",
    "Ice cream", "Lasagna", "Lobster bisque", "Lobster roll sandwich", "Macaroni and cheese",
    "Macarons", "Miso soup", "Mussels", "Nachos", "Omelette", "Onion rings", "Oysters",
    "Pad thai", "Paella", "Pancakes", "Panna cotta", "Peking duck", "Pho", "Pizza",
    "Pork chop", "Poutine", "Prime rib", "Pulled pork sandwich", "Ramen", "Ravioli",
    "Red velvet cake", "Risotto", "Samosa", "Sashimi", "Scallops", "Seaweed salad",
    "Shrimp and grits", "Spaghetti bolognese", "Spaghetti carbonara", "Spring rolls",
    "Steak", "Strawberry shortcake", "Sushi", "Tacos", "Takoyaki", "Tiramisu",
    "Tuna tartare", "Waffles"
]

# Load the trained model
model = load_model('food_model.h5')

# Helper Functions
def preprocess_image(image_path):
    """Preprocesses the image for prediction."""
    image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def decode_predictions(predictions):
    """Decodes the model predictions into a human-readable class label."""
    predicted_index = np.argmax(predictions, axis=1)[0]
    return food_classes[predicted_index]

# Inference Function
def analyze_image(image_path):
    """Analyzes the image and prints the predicted food class."""
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    
    # Print raw predictions for debugging
    print("Predictions (confidence scores):", predictions)
    
    food_item = decode_predictions(predictions)
    print(f"Predicted Food Item: {food_item}")

# Example Usage
image_path = "waffles_0.jpg"  # Replace with your image path
analyze_image(image_path)
