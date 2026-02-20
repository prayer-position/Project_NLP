import pandas as pd

def read_places(path):
    """
    Read csv file TripAdvisor.csv from path
    """
    places = pd.read_csv(path)
    places = places.drop(['idTrip', 'fromId', 'url', 'nbAvisRecupere'], axis = 1)
    return places


def precision_k_typeR(place_id: str, recommendations: tuple[str, float], df) -> float:
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