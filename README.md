# Neural Collaborative Filtering Book Recommendation System
Darrell Silva  
MSDS692 S40 Data Science Practicum

## Introduction
Recommendation systems generate lists of items a user might be interested in. A well known example of this is movie recommendation in Netflix, Hulu, or streaming content providers. Collaborative filtering recommendation systems recommend items based on what the user is interested in compared to what similar users are interested in. In this case, it is based on book ratings. This is able to be accomplished without knowing anything about the the item's content such as genre, actor, or author. Those are used in content filtering recommendation systems and is out of scope for this project. It is useful that you can make recommendations for a user based on other users. However, trouble arises when the item has no ratings, otherwise known as the cold start problem.
  
![collab filtering diagram](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/fca1428cb1223c21aceb25b46d3ce43d7d928362/images/collaborative%20filtering%20diagram.png)

Neural collaborative filtering (NCF) combines neural networks and collaborative filtering based on the user-item interactions. It essentially comes down to user ID, item ID, and the user rating. It takes the user ID's and item ID's as inputs, feeds them through matrix factorization, and then through a Multi-Layer Perceptron Network (MLP). Mapping the user-item ratings along the way.

![ncf diagram](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/4995b3984cb928918e2fa4605f2d15b558a08d2b/images/ncf%20diagram.png)

## The Data

The data used is a subset of the Amazon review data (2018). The primary review data contains all types of items found on Amazon, but only the book data was used. The book review data consisted of 51,311,621 reviews. The review The book metadata contained 2,935,525 books. The compressed file sizes were 11.81 gb review data and 1.22 gb metadata. There were numerous variables in each, but only the book ID, book title, and book category (genre) was imported from the book metadata file while the user ID, book ID, user rating, and review text was imported in the reviews file to reduce computation and storage resource usage.

## Data Ingestion and Wrangling

The project was completed using Python 3.8 in Jupyter Notebooks on a laptop. The laptop ended up being a bottleneck. With limited storage and computing resources, alternate methods were devised to avoid crashing the IDE. Standard methods like a Pandas import were not working so custom functions using the same framework were designed to import data. These functions used the gzip, json, and os libraries to read through the compressed files, pull the desired fields from each row, and load them into a dictionary. For development purposes, row count limiting functionality was also built into it.

```
def load_amazon_review_data(file_name, nrows = 100000000):
    counter = 0
    data = []
    # loading compressed json file
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            # only verified reviews
            if d['verified'] == True:
                # only pulling necessary fields to reduce memory allocation
                d2 = {}
                d2['user_id'] = d.get('reviewerID')
                d2['book_id'] = d.get('asin')
                d2['rating'] = d.get('overall')
                d2['review_text'] = d.get('reviewText')
                data.append(d2)
                counter += 1
            # nrows break
            if (nrows is not None) and (counter > nrows):
                break
    return data
# load data
amazon_reviews = load_amazon_review_data(os.path.join(input_filepath, 'AmazonBooks.json.gz'))
```

Following the raw data import it could be loaded into a Pandas Dataframe.

![raw data merged DF](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/7209d956b9924ae53c81a6ff2bb5f579b1579aaf/images/raw%20data%20merged%20DF.PNG)

After some data exploration and testing, it became apparent the data was still too large. Therefore, book genre and ratings count filtering was implemented. I ended up taking the top ten highest frequency book genres and only using books that had fifty or more ratings. Some additional wrangling was done to the Literature & Fiction genre because it had sub-genres that were the same as other major genres (ex: mystery thriller under literature & fiction being moved the the mystery, thriller, & suspense genre). The remaining literature & fiction sub-genres were labeled as genre fiction.

![genre freqs](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/2e861b7bea6ecc18347fd28da5c5a8872060a6ab/images/genre%20freqs.PNG)

## Exploratory Data Analysis

Given the data consisted of user ID's, book ID's, book titles, and user ratings, an in depth EDA was not necessary. There is the review text but that was earmarked for some extra text analysis functionality which was de-scoped from the project. 

![most reviewed books](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/8622cdb71a2e851c03c7edd33ab0185601d7da88/images/EDA%20most%20reviewed%20books.PNG)

![rating distribution](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/8622cdb71a2e851c03c7edd33ab0185601d7da88/images/EDA%20rating%20distribution.PNG)


+ How
  + EDA: some basic functions and plots
    + Info, describe, value counts, scatter plot, histogram, etc
    + There really isn’t much to explore when your data consists of user ID’s, book ID’s, book titles, and user ratings.
    + EDA wasn’t a big part of the project
  + Modeling:
    + Tensorflow gpu
      + Running into computer resource issues
      + Models were taking a long time running on CPU
      + Google collab throttled/restricted my account within an hour of use
      + Had to adapt and setup jupyter notebooks/tensorflow to run on gpu
      + Neural nets trained much faster and it didn’t consume my CPU/RAM
        + Model training reduced to 6-10 hours from the previous 30 hours.
        + Would run models in the background while at work and overnight
        + Even with GPU processing and filtered data, I was still dealing with 9-18million rows of data.
      + Learned that best practice neural net batch size (32, 64, 128, 256..) because it aligns with how GPUs work and increases processing efficiency/speed
    + Tensorflow functional API method
      + Don’t have to use the functional API method, I just prefer it.
  + Model outputs:
    + Used dimensionality reduction to plot the embeddings developed by the model in 2D
      + PCA – primary component analysis
      + TSNE - t-distributed Stochastic Neighbor Embedding
      + UMAP - Uniform Manifold Approximation and Projection
    + Plotted scatterplots, coloring by the ten book genres
    + Plotted scatterplots, two tone of a single book genre vs the aggregated remaining genres
    + The actual recommender function takes a user ID and outputs the top 5 book recommendations
      + Took the same concept of the dimensionality reduction plots but overlaid the books I’ve read and the recommendations the model made to show how they relate to each other
 
 
 
## References
Ni, J., Li, J., McAuley, J. (2019). Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects. Proceedings of the 2019 Conference on        Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 188–197.   http://dx.doi.org/10.18653/v1/D19-1018 
