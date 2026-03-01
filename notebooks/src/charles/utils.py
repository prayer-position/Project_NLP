import pandas as pd
from pathlib import Path
import urllib.request
import numpy as np
from rank_bm25 import BM25Okapi

def ensure_data(dir, file, url):
    if not file.exists():
        print(f"Data not found at {file}. Dowloading from GitHub")
        dir.mkdir(parents = True, exist_ok = True)

        try:
            urllib.request.urlretrieve(url, file)
            print("Download successful!")
        except Exception as e:
            print(f"Failed to download adata: {e}")
    else: 
        print("Data found locally, skipping download")

def read_places(path):
    """
    Read csv file TripAdvisor.csv from path
    """
    places = pd.read_csv(path)
    places = places.drop(['idTrip', 'fromId', 'url', 'nbAvisRecupere'], axis = 1)
    return places

def dict_to_tuple(original):
    ids = []
    confidences = []
    for item in original:
        ids.append(item.get('id'))
        confidences.append(item.get('confidence'))
    return (ids, confidences)

def lvl_1_eval(place_id: str, recommendations: list[str], df) -> float:
    """
    Measures precision@k of the recommendation using only typR:
    If recommendation typeR is same as original place, then is 1, otherwise is 0
    calculate mean of all this
    
    Parameters: 
        place_id: id of the original place from which we derive the recommendations
        recommendations: tuple:
            value1: id of the recommended place
            value2: trust factor of the recommendation

    Returns:
        Mean of the recommendation typeR
    """
    target_row = df[df['id'] == place_id]
    if target_row.empty:
        return 0.0
    place_typeR = target_row['typeR'].iloc[0]
    correct_matches = 0
    for reco_id in recommendations:
        rec_row = df[df['id'] == reco_id]
        
        if not rec_row.empty:
            rec_typeR = rec_row['typeR'].iloc[0]
            if rec_typeR == place_typeR:
                correct_matches += 1

    return correct_matches / len(recommendations) if recommendations else 0.0

def get_metadata(place_id: str, df: pd.DataFrame, trans_df: pd.DataFrame):
    """
    Extracts interesting categories relation to specific typeR of a place
    Returns a set of categories to simplify comparison
    """
    row = df[df['id'] == place_id]
    type_r = row['typeR'].iloc[0]
    final_set = set()
    path = Path.cwd() / '..' / 'data'

    # Attractions
    if type_r == 'A':
        final_set.add(translate_id(row['activiteSubCategorie'].iloc[0], 'sub_cat', trans_df))
        final_set.add(translate_id(row['activiteSubType'].iloc[0], 'sub_type', trans_df))

    # Restaurants
    elif type_r == 'R':
        final_set.add(translate_id(row['restaurantTypeCuisine'].iloc[0], 'cuisine', trans_df))
        final_set.add(translate_id(row['restaurantDietaryRestrictions'].iloc[0], 'diet_restric', trans_df))
        final_set.add(translate_id(row['restaurantType'].iloc[0], 'rest_type', trans_df))

    # Hotels
    elif type_r == 'H':
        price = row['priceRange'].iloc[0]
        if pd.notna(price):
            final_set.add(str(price).strip().lower())

    return {str(cat).lower() for cat in final_set if cat}

def translate_id(raw_id: str, prefix: str, df: pd.DataFrame):
    """
    Safely translates a single ID.
    Returns the translated name as a string, or None if not found/missing.
    """
    if pd.isna(raw_id) or str(raw_id).lower() == 'none':
        return None

    search_id = f"{prefix}_{raw_id}"
    match = df[df['id'] == search_id]

    if not match.empty:
        return match['name'].iloc[0]
    
    return None

def get_translation_dicts():
    """
    Reads the csvs for the categories/types/etc. to avoid reading them every call of the lvl_2_eval function
    """
    df = pd.DataFrame()
    path = Path.cwd() / '..' / 'data'

    files = {
    'sub_cat': 'AttractionSubCategorie.csv', 
    'sub_type': 'AttractionSubType.csv',
    'cuisine': 'cuisine.csv',
    'diet_restric': 'dietary_restrictions.csv',
    'rest_type': 'restaurantType.csv',
    }

    for prefix, file_name in files.items():
        temp_df = pd.read_csv(path / file_name)
        # Add a prefix to the ID: e.g., "20" becomes "sub_cat_20"
        temp_df['id'] = prefix + "_" + temp_df['id'].astype(str)
        df = pd.concat([df, temp_df])

    return df

def lvl_2_eval(place_id: str, recommendations: list[str], df: pd.DataFrame, translation_df: pd.DataFrame):
    """
    Parameters: 
        place_id: id of the original place from which we derive the recommendations
        recommendations: id of the recommended places
    
    Returns score based on ranking
    """
    query_meta = get_metadata(place_id, df, translation_df)

    for rank, recomendation_id in enumerate(recommendations):
        recommendation_meta = get_metadata(recomendation_id, df, translation_df)
        for data in query_meta:
            if data in recommendation_meta:
                return rank
            
    return None

