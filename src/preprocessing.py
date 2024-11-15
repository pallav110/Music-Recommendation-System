import pandas as pd

# Emotion to musical feature mapping
emotion_feature_mapping = {
    'Anxiety': ['Slow Tempo', 'Soft Consonant Harmony'],
    'Sadness': ['Dynamics', 'Moderate Tempo', 'Ascending Melody', 'Major Harmony'],
    'Anger': ['Dynamic Rhythm', 'Complex Harmony', 'High Intensity'],
    'Fear': ['Soft Tempo', 'Gentle Repetitive Melody'],
    'Loneliness': ['Rhythm', 'Warm Timbre', 'Moderate Tempo', 'Nostalgic Themes'],
    'Frustration': ['Steady Rhythm', 'Minor Harmony', 'Low Pitch'],
    'Melancholy': ['Slow Tempo', 'Melancholic Harmony', 'Reflective Melody'],
    'Guilt': ['Slow Tempo', 'Soft Dynamics', 'Warm Timbre'],
    'Joy': ['Fast Tempo', 'Major Harmony', 'Ascending Melody'],
    'Excitement': ['Fast Tempo', 'Complex Rhythm', 'Bright Timbre'],
    'Calm': ['Soft Tempo', 'Repetitive Melody', 'Gentle Rhythm'],
    'Love': ['Warm Timbre', 'Moderate Tempo', 'Consonant Harmony'],
    'Optimism': ['Moderate Tempo', 'Consonant Harmony', 'Balanced Dynamics'],
    'Grief': ['Fast Tempo', 'Major Harmony', 'Bright Timbre'],
    'Serenity': ['Slow Tempo', 'Reflective Melody', 'Melancholic Harmony']
}

def load_data(filepath):
    """
    Load the dataset from the given CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def map_emotion_to_song(features_df):
    """
    Map the emotion to each song based on its features.
    """
    emotion_column = []
    
    for index, row in features_df.iterrows():
        tempo = row['Tempo']
        energy = row['Energy']
        dynamics = row['Energy']  # Energy can be used for dynamics
        harmony = row.get('Mode', '')  # Mode for harmony
        rhythm = row.get('Danceability', '')  # Danceability for rhythm
        
        # Example logic for emotion mapping based on features
        if tempo < 80 and 'Soft Consonant Harmony' in emotion_feature_mapping['Anxiety']:
            emotion_column.append('Anxiety')
        elif 80 <= tempo <= 120 and 'Moderate Tempo' in emotion_feature_mapping['Sadness']:
            emotion_column.append('Sadness')
        elif energy > 0.7 and 'Dynamic Rhythm' in emotion_feature_mapping['Anger']:
            emotion_column.append('Anger')
        elif tempo < 80 and 'Gentle Repetitive Melody' in emotion_feature_mapping['Fear']:
            emotion_column.append('Fear')
        elif 120 <= tempo <= 140 and 'Moderate Tempo' in emotion_feature_mapping['Loneliness']:
            emotion_column.append('Loneliness')
        elif energy < 0.4 and 'Steady Rhythm' in emotion_feature_mapping['Frustration']:
            emotion_column.append('Frustration')
        elif tempo < 80 and 'Melancholic Harmony' in emotion_feature_mapping['Melancholy']:
            emotion_column.append('Melancholy')
        elif tempo < 80 and 'Soft Dynamics' in emotion_feature_mapping['Guilt']:
            emotion_column.append('Guilt')
        elif tempo > 120 and 'Major Harmony' in emotion_feature_mapping['Joy']:
            emotion_column.append('Joy')
        elif tempo > 120 and 'Complex Rhythm' in emotion_feature_mapping['Excitement']:
            emotion_column.append('Excitement')
        elif tempo < 80 and 'Gentle Rhythm' in emotion_feature_mapping['Calm']:
            emotion_column.append('Calm')
        elif 'Warm Timbre' in emotion_feature_mapping['Love']:
            emotion_column.append('Love')
        elif tempo > 100 and 'Balanced Dynamics' in emotion_feature_mapping['Optimism']:
            emotion_column.append('Optimism')
        elif tempo > 100 and 'Grief Support' in emotion_feature_mapping['Grief']:
            emotion_column.append('Grief')
        elif tempo < 80 and 'Reflective Melody' in emotion_feature_mapping['Serenity']:
            emotion_column.append('Serenity')
        else:
            emotion_column.append('Unknown')  # Default if no match

    features_df['Emotion'] = emotion_column
    return features_df

def preprocess_data(filepath, output_filepath):
    """
    Preprocess the data by merging features and mapping emotions, and save to CSV.
    """
    df = load_data(filepath)
    
    # Keep necessary columns (Track Name, Artist Name, and Album)
    columns_to_drop = ['Track URI', 'Artist URI']
    df = df.drop(columns=columns_to_drop)
    
    # Map emotion based on features
    processed_df = map_emotion_to_song(df)
    
    # Save the processed data with emotions to CSV, including Track Name, Artist Name, and Album
    processed_df.to_csv(output_filepath, index=False)
    print(f"Processed data saved to {output_filepath}")

# Example usage
if __name__ == '__main__':
    input_filepath = 'data/hindi_songs.csv'  # Replace with your input CSV path
    output_filepath = 'data/clean.csv'  # Replace with your desired output CSV path
    preprocess_data(input_filepath, output_filepath)
