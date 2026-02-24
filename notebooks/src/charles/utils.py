import pandas as pd
from pathlib import Path

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

def lvl_1_eval(place_id: str, recommendations: list[tuple[str, float]], df) -> float:
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
        Mean of the recommendation typeR with regard to trust factor
    """
    place_typeR = df[df['id'] == place_id]['typeR'].iloc[0]
    sum = 0
    max_sum = 0
    recommendation_ids, tfs = recommendations
    for recommendation_id, tf in zip(recommendation_ids, tfs):
        recommendation_typeR = df[df['id'] == recommendation_id]['typeR'].iloc[0]
        if recommendation_typeR == place_typeR  :
            sum += tf
        max_sum += tf
    return sum/max_sum

def get_metadata(place_id, df):
    """
    Extracts interesting categories relation to specific typeR of a place
    Returns a set of categories to simplify comparison
    """
    row = df[df['id'] == place_id].iloc[0]
    type_r = row['typeR']
    
    categories = []
    path = Path.cwd() / '..' / 'data'

    # Attractions
    if type_r in ['A', 'AP']:
        categories.append(translate_id(row['activiteSubCatecorie'], path / 'AttratctionSubCategorie.csv'))
        categories.append(translate_id(row['activiteSubType'], path / 'AttractionSubType.csv'))

    # Restaurants
    elif type_r == 'R':
        categories.append(translate_id(row['restaurantTypeCuisine'], path / 'cuisine.csv'))
        categories.append(translate_id(row['restaurantDietaryRestrtictions'], path / 'dietary_restrictions.csv'))
        categories.append(translate_id(row['restaurantType'], path / 'restaurantType.csv'))

    # Hotels
    elif type_r == 'H':
        categories.append(row['priceRange'])

    final_set = set()
    for cat in categories:
        if pd.notna(cat) and str(cat).lower() != 'none':
            if isinstance(cat, str) and ',' in cat:
                parts = [p.strip().lower() for p in cat.split(',')]
                final_set.update(parts)
            else:
                final_set.add(str(cat.strip().lower()))
    
    return final_set

def translate_id(id, path):
    """
    Translates id from the dataset into a word
    Parameters: 
        id: id of the type/cuisine/categorie
        path: path of the file we want to read
    """
    df = pd.read_csv(path)
    return df[df['id'] == id]['name'].iloc[0]

def lvl_2_eval(place_id: str, recommendations: list[tuple[str, float]], df: pd.DataFrame):
    """
    Parameters: 
        place_id: id of the original place from which we derive the recommendations
        recommendations: id of the recommended places
    
    Returns score based on ranking
    """
    query_emta = get_metadata(place_id)

    for rank, recomendation_id in enumerate(recommendations):
        recommendation_meta = get_metadata(recomendation_id)