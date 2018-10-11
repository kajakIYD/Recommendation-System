import numpy as np
from lightfm.datasets import  fetch_movielens
from lightfm import LightFM

def sample_recommedation(model, data, user_ids):

    # Number of users and movies in training data
    n_users, n_items = data['train'].shape

    # Generate recommendation for each user
    # in collaborative+content_based way
    for user_id in user_ids:

        # Already liked movies by user
        liked_movies = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # Movies predicted as to-be-liked
        predicted_liked_movies = model.predict(user_id, np.arange(n_items))

        # Rank predictions in order from most to least to-be-liked
        top_items = data['item_labels'][np.argsort(-predicted_liked_movies)]

        # Print results
        print('*' * 20)
        print('User %s  has following already liked movies:' % user_id)
        for title in liked_movies[:2]:
            print(title)

        print('And following movies suggested as to-be-liked:')
        for title in top_items[:2]:
            print(title)


# Acquire data with specified rating
data = fetch_movielens(min_rating=4.5)

print(repr(data['train']))
print(repr(data['test']))

#print(str(data['train']))
#print(str(data['test']))

# Create model
model_bpr = LightFM(loss='bpr')
model_warp = LightFM(loss='warp')
models = [model_bpr, model_warp]

# Train model
for model in models:
    model.fit(data['train'], epochs=40, num_threads=2)
    sample_recommedation(model, data, [2, 3, 5])
