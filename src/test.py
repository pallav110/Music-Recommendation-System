import joblib
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline
import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
import xgboost as xgb

# Emotion to features mapping (all lowercase for consistency)
# Map emotion features to dataset columns
feature_mapping = {
    'Slow Tempo': ['Tempo'],
    'Moderate Tempo': ['Tempo'],
    'Fast Tempo': ['Tempo'],
    'Dynamics': ['Loudness', 'Energy'],  # Could relate to Loudness and Energy
    'Ascending Melody': ['Danceability'],  # Danceability could relate to melody direction
    'Major Harmony': ['Key', 'Mode'],  # Major Harmony might map to Key and Mode
    'Minor Harmony': ['Key', 'Mode'],  # Minor Harmony maps to Key and Mode
    'Dynamic Rhythm': ['Energy'],  # High-energy rhythm could map to Energy
    'Complex Harmony': ['Key', 'Mode'],  # Complex Harmony maps to Key and Mode
    'High Intensity': ['Energy'],  # High intensity relates to Energy
    'Gentle Repetitive Melody': ['Danceability'],  # Melody + Repetition could be Danceability
    'Rhythm': ['Danceability', 'Energy'],  # Rhythm could relate to Danceability and Energy
    'Warm Timbre': ['Speechiness', 'Acousticness'],  # Timbre could relate to Speechiness and Acousticness
    'Steady Rhythm': ['Danceability'],  # Rhythm relates to Danceability
    'Reflective Melody': ['Danceability'],  # Reflective Melody could be associated with Danceability
    'Consonant Harmony': ['Key', 'Mode'],  # Consonant Harmony relates to Key and Mode
    'Balanced Dynamics': ['Energy', 'Loudness'],  # Dynamics relate to Energy and Loudness
    'Bright Timbre': ['Speechiness', 'Acousticness'],  # Timbre could be linked to Speechiness and Acousticness
    'Nostalgic Themes': ['Acousticness'],  # Nostalgic Themes might correlate with Acousticness
    'Soft Dynamics': ['Loudness', 'Energy'],  # Soft dynamics likely link to Loudness and Energy
    'Melancholic Harmony': ['Key', 'Mode'],  # Melancholic Harmony maps to Key and Mode
    'Repetitive Melody': ['Danceability'],  # Melody repetition relates to Danceability
    'Gentle Rhythm': ['Tempo', 'Energy'],  # Gentle Rhythm relates to Tempo and Energy
    'Reflective Melody': ['Danceability']  # Reflective Melody could be Danceability
}

# Adjust the emotion_to_features dictionary
emotion_to_features = {
    'anxiety': ['Slow Tempo', 'Soft Consonant Harmony'],
    'sadness': ['Dynamics', 'Moderate Tempo', 'Ascending Melody', 'Major Harmony'],
    'anger': ['Dynamic Rhythm', 'Complex Harmony', 'High Intensity'],
    'fear': ['Soft Tempo', 'Gentle Repetitive Melody'],
    'loneliness': ['Rhythm', 'Warm Timbre', 'Moderate Tempo', 'Nostalgic Themes'],
    'frustration': ['Steady Rhythm', 'Minor Harmony', 'Low Pitch'],
    'melancholy': ['Slow Tempo', 'Melancholic Harmony', 'Reflective Melody'],
    'guilt': ['Slow Tempo', 'Soft Dynamics', 'Warm Timbre'],
    'joy': ['Fast Tempo', 'Major Harmony', 'Ascending Melody'],
    'excitement': ['Fast Tempo', 'Complex Rhythm', 'Bright Timbre'],
    'calm': ['Soft Tempo', 'Repetitive Melody', 'Gentle Rhythm'],
    'love': ['Warm Timbre', 'Moderate Tempo', 'Consonant Harmony'],
    'optimism': ['Moderate Tempo', 'Consonant Harmony', 'Balanced Dynamics'],
    'grief': ['Fast Tempo', 'Major Harmony', 'Bright Timbre'],
    'serenity': ['Slow Tempo', 'Reflective Melody', 'Melancholic Harmony'],
    'neutral': ['Moderate Tempo', 'Balanced Dynamics', 'Soft Melody']
}


# Initialize the emotion classification pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1 , device='cuda')

def load_data(filepath):
    # Load and preprocess your dataset
    return pd.read_csv(filepath)

def analyze_emotion(conversation_text):
    # Use the emotion classifier to predict the emotion of the input text
    result = emotion_classifier(conversation_text)
    if result:
        detected_emotion = result[0][0]['label'].lower()  # Ensure the detected emotion is lowercase
        return detected_emotion  
    # Normalize to match other parts of the code
    else:
        return None
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd

def recommend_songs(emotion, df, model, le):
    # Ensure the emotion is part of the predefined emotions
    emotion = emotion.lower()  # Ensure the input emotion is lowercase
    print(f"Detected emotion: {emotion}")  # Print the detected emotion

    # Check if the emotion exists in the emotion_to_features mapping
    if emotion not in emotion_to_features:
        detected_features = emotion_to_features['neutral']  # Set to a default like 'neutral' if emotion not found
        emotion = 'neutral'
    else:
        detected_features = emotion_to_features[emotion]

    print("Detected Features:", detected_features)
    print("Length of Detected Features:", len(detected_features))

    # Initialize a LabelEncoder for converting features into numerical values
    feature_encoder = LabelEncoder()
    encoded_features = feature_encoder.fit_transform(detected_features)

    print("Encoded Features:", encoded_features)

    # Ensure that we have exactly 10 features in the DataFrame, pad with default values (e.g., 0 or NaN)
    feature_columns = ['Danceability', 'Energy', 'Tempo', 'Loudness', 'Mode', 
                       'Key', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness']
    
    # Create a list of encoded features and pad with default values
    feature_data = encoded_features.tolist() + [0] * (len(feature_columns) - len(encoded_features))

    # Create the DataFrame with the expected number of columns
    feature_df = pd.DataFrame([feature_data], columns=feature_columns)

    print("Feature DataFrame:\n", feature_df)

    # Predict using the model with the encoded features
    dmatrix = xgb.DMatrix(feature_df)
    predicted_output = model.predict(dmatrix)

    # Check columns in the df to verify 'Emotion'
    print("Columns in the dataframe:", df.columns)

    # Convert the Emotion column to lowercase for consistency
    df['Emotion'] = df['Emotion'].str.strip().str.lower()  # Strip spaces and convert to lowercase

    # Debug: Print unique emotions in the DataFrame
    print("Unique emotions in the dataframe:", df['Emotion'].unique())

    # Filter the DataFrame based on detected emotion and features
    df_filtered = df[df['Emotion'] == emotion]

    # Debug: Check how many rows match the detected emotion
    print(f"Rows matching emotion '{emotion}': {len(df_filtered)}")

    # Map the detected features to the relevant columns in the DataFrame
    mapped_features = []
    for feature in detected_features:
        mapped_features.extend(feature_mapping.get(feature, []))  # Map each feature to corresponding dataset columns

    print("Mapped Features:", mapped_features)

    # Further filter songs based on mapped features (matching non-zero values in relevant columns)
    recommended_songs = df_filtered[df_filtered[mapped_features].apply(
        lambda x: any(val > 0 for val in x), axis=1
    )]

    # Debug: Check the recommended songs
    print("Recommended Songs:\n", recommended_songs)

    # Return top 5 recommended songs
    return recommended_songs.head(5)

def predict_and_display():
    conversation_text = chat_log.get("1.0", tk.END).strip()
    if not conversation_text:
        messagebox.showinfo("Info", "Please chat to analyze your emotion.")
        return
    
    detected_emotion = analyze_emotion(conversation_text)
    if not detected_emotion:
        messagebox.showinfo("Error", "Emotion analysis failed.")
        return

    messagebox.showinfo("Detected Emotion", f"Detected emotion: {detected_emotion}")

    # Load data and model
    input_filepath = 'data/clean.csv'  # Adjust path as necessary
    df = load_data(input_filepath)
    model = joblib.load('best_xgb_model.pkl')  # Adjust path as necessary
    le = LabelEncoder()  # Adjust path as necessary

    recommended_songs = recommend_songs(detected_emotion, df, model, le)

    if not recommended_songs.empty:
        result = "\n".join(recommended_songs['Track_Name'].tolist())
        messagebox.showinfo("Recommended Songs", f"Songs for {detected_emotion.lower()} emotion:\n{result}")
    else:
        messagebox.showinfo("No Recommendations", f"No songs found for the emotion: {detected_emotion.lower()}.")

# Create a simple chat interface using Tkinter
root = tk.Tk()
root.title("Chat Interface with Emotion Analysis")

chat_log = tk.Text(root, height=15, width=50)
chat_log.pack()

predict_button = tk.Button(root, text="Analyze and Recommend Songs", command=predict_and_display)
predict_button.pack()

root.mainloop()
