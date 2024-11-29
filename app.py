from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import google.generativeai as genai

app = Flask(__name__)

# Nutritionix API credentials
nutritionix_app_id = "your_id"
nutritionix_app_key = "your_key"


# Configure Gemini API with your API key
genai.configure(api_key="your_api_key")


UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('food_model.h5')

# Food classes
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

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def fetch_nutrition_info(food_item, quantity=None):
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {
        "x-app-id": nutritionix_app_id,
        "x-app-key": nutritionix_app_key,
        "Content-Type": "application/json"
    }
    query = {"query": food_item}
    if quantity:
        query["query"] += f" {quantity} grams"
    response = requests.post(url, headers=headers, json=query)
    return response.json() if response.status_code == 200 else {"error": "Unable to fetch nutrition data"}

def generate_allergen_message(allergies, food_name):
    """
    Use GenAI to generate a user-friendly message about allergens.
    """
    prompt = (f"You are an assistant that provides allergen alerts for users based on their dietary restrictions. The user has consumed a {food_name}, and they have specified certain allergens they are sensitive to those are {allergies}. If the allergens are found in the food item, generate a detailed alert message. Highlight the allergens found and emphasize the potential health risks, urging caution. If the allergens are NOT found in the food item, generate a reassuring message confirming the food is safe for the user to consume based on their specified allergens. Keep the messages concise, clear, and empathetic.")

    try:
        model = genai.GenerativeModel("gemini-1.5-flash") 
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the allergen message: {e}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/nutrition-analysis', methods=['GET', 'POST'])
def nutrition_analysis():
    if request.method == 'POST':
        image = request.files['food-image']
        weight = request.form.get('food-weight', None)
        allergies = request.form.get('allergies', "").lower()

        # Save image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        # Predict food
        processed_image = preprocess_image(image_path)
        predictions = model.predict(processed_image)
        predicted_food = food_classes[np.argmax(predictions, axis=1)[0]]

        # Fetch nurition info
        nutrition_data = fetch_nutrition_info(predicted_food, weight)
        os.remove(image_path)

        if "error" in nutrition_data:
            return render_template('result.html', analysis_type='Nutrition Analysis', result={
                'error': nutrition_data["error"]
            })

        nutrition = nutrition_data["foods"][0]
        important_nutrients = {
            "Food Name": nutrition.get("food_name", "N/A"),
            "Measured Quantity (g)": nutrition.get("serving_weight_grams", "N/A"),
            "Calories": nutrition.get("nf_calories", "N/A"),
            "Total Fat (g)": nutrition.get("nf_total_fat", "N/A"),
            "Saturated Fat (g)": nutrition.get("nf_saturated_fat", "N/A"),
            "Trans Fat (g)": nutrition.get("trans_fat", "N/A"),
            "Cholesterol (mg)": nutrition.get("nf_cholesterol", "N/A"),
            "Sodium (mg)": nutrition.get("nf_sodium", "N/A"),
            "Total Carbohydrates (g)": nutrition.get("nf_total_carbohydrate", "N/A"),
            "Dietary Fiber (g)": nutrition.get("nf_dietary_fiber", "N/A"),
            "Sugars (g)": nutrition.get("nf_sugars", "N/A"),
            "Protein (g)": nutrition.get("nf_protein", "N/A"),
            "Potassium (mg)": nutrition.get("nf_potassium", "N/A"),
        }

        if allergies.strip(): 
            allergen_message = generate_allergen_message(allergies, predicted_food)
        else:
            allergen_message = "No allergens specified by the user. No allergen check performed."

        return render_template('result.html', analysis_type='Nutrition Analysis', result={
           'predicted_food': predicted_food,
           'nutrition_info': important_nutrients,
           'allergy_alert': allergen_message
})

    return render_template('nutrition.html')

@app.route('/calories-analysis', methods=['GET', 'POST'])
def calories_analysis():
    if request.method == 'POST':
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        age = int(request.form['age'])
        gender = request.form['gender']

        if gender == "male":
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        else:
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

        tdee = bmr * 1.55

        return render_template('result.html', analysis_type='Calories Analysis', result={
            'bmr': round(bmr, 2),
            'tdee': round(tdee, 2)
        })
    return render_template('calories.html')

if __name__ == '__main__':
    app.run(debug=True)
