import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load Skincare Dataset
dff = pd.read_csv("result.csv")

# Acne Subtype to Ingredients Mapping
acne_subtype_to_ingredients = {
    'blackheads': ['salicylic acid', 'retinoids', 'benzoyl peroxide'],
    'whiteheads': ['salicylic acid', 'retinoids', 'AHA', 'BHA'],
    'cysts': ['benzoyl peroxide', 'sulfur', 'retinoids'],
    'papules': ['benzoyl peroxide', 'salicylic acid', 'sulfur'],
    'pustules': ['benzoyl peroxide', 'salicylic acid', 'sulfur', 'antibiotics']
}


# Recommend Products Based on Acne Type and Skin Type
def recommend_products_based_on_acne_type(acne_type, dff):
    # Get ingredients based on acne type
    ingredients_needed = acne_subtype_to_ingredients.get(acne_type.lower())
    if not ingredients_needed:
        return "Invalid acne type"

    recommendations = []
    for _, row in dff.iterrows():
        key_ingredient = row['key ingredient']
        product_description = row['product description'] if 'product description' in row else ''
        

        if isinstance(key_ingredient, str) and any(ingredient in key_ingredient.lower() for ingredient in ingredients_needed):
            recommendations.append(row)
    
    if not recommendations:
        return "No products found for this acne type."
    
    recommended_df = pd.DataFrame(recommendations)
    return recommended_df[['brand', 'name', 'url', 'skin type', 'concern']]

