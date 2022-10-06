# Neural Collaborative Filtering Book Recommendation System
Darrell Silva  
MSDS692 S40 Data Science Practicum

## Introduction
  Recommendation systems generate lists of items a user might be interested in. A well known example of this is movie recommendation in Netflix, Hulu, or streaming content providers.  
    Collaborative filtering recommendation systems recommend items based on what the user is interested in compared to what similar users are interested in. In this case, it is based on book ratings. This is able to be accomplished without knowing anything about the the item's content such as genre, actor, or author. Those are used in content filtering recommendation systems and is out of scope for this project. It is useful that you can make recommendations for a user based on other users. However, trouble arises when the item has no ratings, otherwise known as the cold start problem.
  
![collab filtering diagram](https://github.com/DarrellS0352/msds692_s40_data_science_practicum_1/blob/fca1428cb1223c21aceb25b46d3ce43d7d928362/images/collaborative%20filtering%20diagram.png)

  + Neural collaborative filtering (NCF) combines neural networks and collaborative filtering based on the user-item interactions. It essentially comes down to user ID, item ID, and the user rating. It takes the user ID's and item ID's as inputs, feeds them through matrix factorization, and then through a Multi-Layer Perceptron Network (MLP). Mapping the user-item ratings along the way.
+ When
  + Proposed project timeline
    + Week 1 Project proposal
    + Week 2 Data ingestion and wrangling
    + Week 3 EDA and data cleaning
    + Week 4 Model building, visualizations, model outputs (recommendations)
    + Week 5/6 reconsidering bells and whistles to improve project (cloud/visualization tools/web scraping/review text analysis/etc)
    + Week 7 Final touches and working on presentation
    + Week 8 Working on presentation and presenting
  + Actual project timeline
    + Week 1 Project proposal, data ingestion and wrangling
    + Week 2 Data ingestion, wrangling, and modeling iteratively
    + Week 3 Data wrangling, modeling, and model output development iteratively
    + Week 4 Data wrangling, modeling, and model output development iteratively
    + Week 5 Data wrangling, modeling, and model output development iteratively
    + Week 6 Data wrangling, modeling, and model output development iteratively
    + Week 7 Wrapping up modeling, code, and development. Start project deliverables.
    + Week 8 Finish project deliverables. Presentation.
+ How
  + High level: python, jupyter notebooks, laptop
  + Data: where I got the data from
  + Data ingestion/wrangling: gzip, json, and pandas
    + Files were so large I kept crashing google chrome and jupyter notebooks
    + Had limited hard drive space
    + Had to filter data to smaller sizes for project feasibility
    + Initially started with data from Goodreads and Amazon
    + Goodreads book categories were very messy, and I quickly realized cleaning them to a usable standard would take too long for project feasibility
    + Stopped using Goodreads data
    + Tried random sampling
    + Eventually decided to use the top 10 most popular genres
    + Had to do some extra wrangling because Literature & Fiction genre had sub-genres that overlap with other major genres
    + Kept having to iterate so I setup the code where I would only have to change which book genres to select or add minimal additional code and run the whole notebook in order to produce a new file
    + Manual book genre filtering/selection through iteration and most useful
    + Import book data, wrangle, clean
    + Import review data
    + Merge books and reviews
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
