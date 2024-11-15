import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # Progress bar library
import numpy as np
import joblib  # To save the trained model
from sklearn.preprocessing import LabelEncoder  # To encode the labels

# Load your cleaned CSV data
def load_data(filepath):
    """
    Load the dataset from the cleaned CSV file.
    """
    df = pd.read_csv(filepath)
    return df

# Train XGBoost Model with GPU and Progress Bar
def train_xgb(features, target):
    """
    Train the XGBoost model on the features and target with progress bar.
    """
    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Convert the datasets into DMatrix, which is a data structure optimized for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set XGBoost parameters (adjust based on your dataset)
    params = {
        'objective': 'multi:softmax',  # Multi-class classification
        'num_class': len(np.unique(target)),  # Number of unique emotions
        'eval_metric': 'merror',  # Use multi-class error metric
        'max_depth': 6,  # Maximum depth of the tree
        'learning_rate': 0.05,  # Learning rate
        'subsample': 0.8,  # Fraction of samples used for training each tree
        'colsample_bytree': 0.8,  # Fraction of features used for each tree
        'tree_method': 'hist',  # Use 'hist' method for faster training
        'device': 'cuda',  # Use GPU for training
    }

    # Initialize the best accuracy variable
    best_accuracy = 0

    # Training the model with progress bar for epochs
    num_epochs = 1000  # Max epochs, can be adjusted
    early_stopping_rounds = 50  # Stop training after 50 rounds without improvement

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        # Train the model with early stopping
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,  # Max number of boosting rounds (trees)
            evals=[(dtest, 'eval')],
            early_stopping_rounds=early_stopping_rounds,  # Early stopping after 50 rounds
            verbose_eval=False,  # Disable verbose output
        )

        # Make predictions and evaluate the model
        y_pred = model.predict(dtest)
        y_pred_max = [int(pred) for pred in y_pred]  # Get the class with the highest probability

        accuracy = accuracy_score(y_test, y_pred_max)

        # If accuracy improves, save the model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print(f'Best Accuracy: {best_accuracy:.4f}')

    # Save the best model
    joblib.dump(best_model, 'best_xgb_model.pkl')

    return best_model, best_accuracy

# Function to recommend songs based on emotion
def recommend_songs(user_emotion, df, model, le):
    """
    Recommend songs based on the user's desired emotion.
    """
    # Check if the user's emotion is in the LabelEncoder's classes
    if user_emotion not in le.classes_:
        print(f"Emotion '{user_emotion}' is not recognized. Please enter a valid emotion.")
        return pd.DataFrame()  # Return an empty dataframe if emotion is not valid

    # Get the emotion label from the user input
    emotion_label = le.transform([user_emotion])[0]

    # Predict the emotion for each song in the dataset
    features = df[['Danceability', 'Energy', 'Tempo', 'Loudness', 'Mode', 'Key', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness']]
    dfeatures = xgb.DMatrix(features)
    predictions = model.predict(dfeatures)

    # Map the predictions back to emotion labels
    emotion_predictions = le.inverse_transform(predictions.astype(int))

    # Add the predicted emotions to the dataframe
    df['Predicted_Emotion'] = emotion_predictions

    # Filter the songs based on the user’s emotion preference
    recommended_songs = df[df['Predicted_Emotion'] == user_emotion]

    return recommended_songs

# Main function to load data, train model, and evaluate
def main():
    input_filepath = 'data/clean.csv'  # Path to your preprocessed cleaned CSV
    df = load_data(input_filepath)

    # Assuming 'Emotion' is the target column
    features = df[['Danceability', 'Energy', 'Tempo', 'Loudness', 'Mode', 'Key', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness']]

    # Label encode the target column (Emotion)
    le = LabelEncoder()
    target = le.fit_transform(df['Emotion'])

    # Train the model and evaluate
    # model, accuracy = train_xgb(features, target)
    # print(f'Model trained with final accuracy: {accuracy:.4f}')
    model = joblib.load('best_xgb_model.pkl')
    # Ask the user for their desired emotion
    user_emotion = input("Enter the emotion you want (e.g., 'happy', 'sad', 'angry', etc.): ")

    # Get the recommended songs based on the user’s emotion
    recommended_songs = recommend_songs(user_emotion, df, model, le)

    if not recommended_songs.empty:
        print(f"\nRecommended Songs for {user_emotion.capitalize()} emotion:")
        print(recommended_songs[['Track_Name', 'Predicted_Emotion']])  # Assuming 'Song_Name' is a column in the dataframe
    else:
        print(f"No songs found for the emotion: {user_emotion.capitalize()}.")

if __name__ == '__main__':
    main()
