import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the best-trained Random Forest model
with open('best_random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the dataset
track_data = pd.read_csv('songs_with_moods_ML_Dataset.csv')

# Encode mood labels
mood_labels = ['Calm', 'Excited', 'Happy', 'Melancholic', 'Neutral', 'Relaxed', 'Sad', 'Thoughtful', 'Upset', 'Worried']
label_encoder = LabelEncoder()
label_encoder.fit(mood_labels)

def get_tracks_by_mood(mood_input, track_data, label_encoder):
    try:
        # Encode the mood input
        mood_encoded = label_encoder.transform([mood_input])[0]
        print(f"Encoded mood input: {mood_encoded}")
        
        # Check if the encoded mood is present in the dataset
        if mood_encoded not in track_data['mood'].map(lambda x: label_encoder.transform([x])[0]).values:
            print("The mood input does not match any entries in the dataset.")
            return pd.DataFrame(columns=['track_id', 'title', 'track_popularity'])
        
        # Filter tracks by the input mood
        filtered_tracks = track_data[track_data['mood'] == mood_input]
        
        # Check if filtering worked
        if filtered_tracks.empty:
            print("No tracks found for the specified mood.")
        
        # Select relevant columns
        result = filtered_tracks[['track_id', 'title', 'track_popularity']]
        
        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(columns=['track_id', 'title', 'track_popularity'])

def get_random_song_for_mood(track_data, mood_input, label_encoder):
    # Get tracks by the specified mood
    filtered_tracks = get_tracks_by_mood(mood_input, track_data, label_encoder)
    
    if filtered_tracks.empty:
        print("No tracks found for the specified mood.")
        return pd.DataFrame(columns=['track_id', 'title'])
    
    # Randomly select one song from the filtered tracks
    random_song = filtered_tracks.sample(n=10, random_state=np.random.randint(0, 1000))
    
    return random_song[['track_id', 'title']]

def predict_songs(mood_input):
    # Get random songs for the specified mood
    random_songs = get_random_song_for_mood(track_data, mood_input, label_encoder)
    return random_songs

if __name__ == "__main__":
    # Example usage
    mood_input = 'Happy'  # Change this to the desired mood
    recommendations = predict_songs(mood_input)
    print("Recommended songs:")
    print(recommendations)
