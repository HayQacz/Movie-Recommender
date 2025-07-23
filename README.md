# **Movie Recommendation System**

## **🌟 Overview**

Welcome to the Movie Recommendation System project\! This initiative dives deep into the exciting realm of personalized content discovery. Our goal is to build an intelligent system that suggests movies tailored to individual user preferences, leveraging a variety of machine learning techniques. This repository serves as a comprehensive demonstration of the end-to-end process of building a recommender system, from raw data processing and exploratory analysis to model implementation, evaluation, and future deployment considerations.

## **📊 Dataset**

This project is powered by the **MovieLens 1M Dataset**, a widely recognized benchmark in the recommender systems community. It provides a rich foundation for our analysis and model training, containing:

* **1 million anonymous ratings** provided by 6,040 users across approximately 3,900 movies.  
* Detailed user demographic information, including gender, age, occupation, and zip-code.  
* Comprehensive movie metadata, such as titles and genres.

You can download the dataset directly from the [MovieLens website](https://grouplens.org/datasets/movielens/1m/).

## **📁 Project Structure**

Our project is meticulously organized for clarity and ease of navigation:  
movie\_recommendation\_system/  
├── data/  
│   ├── raw/                \# Original, untouched MovieLens 1M dataset files.  
│   │   └── ml-1m/  
│   └── processed/          \# Cleaned, transformed, and ready-to-use dataframes.  
├── notebooks/  
│   ├── 01\_data\_exploration\_preprocessing.ipynb  \# Comprehensive notebook for EDA and data preparation.  
│   ├── 02\_model\_training\_evaluation.ipynb       \# Core notebook for implementing and evaluating ML models.  
├── src/                    \# (Future) Modular Python scripts for reusable functions and advanced model definitions.  
│   ├── utils.py  
│   └── models.py  
├── requirements.txt        \# Lists all essential Python package dependencies.  
└── README.md               \# The documentation you are currently reading\!

## **🛠️ Technologies Used**

This project leverages a powerful stack of Python libraries:

* **Python:** The foundational programming language.  
* **Pandas:** Indispensable for efficient data manipulation and analysis.  
* **NumPy:** Crucial for high-performance numerical operations.  
* **Scikit-learn:** Provides robust machine learning utilities, including similarity calculations and data splitting.  
* **Matplotlib & Seaborn:** Utilized for creating insightful and visually appealing data visualizations.  
* **Jupyter Notebooks:** Our interactive environment for development, experimentation, and documenting the entire workflow.

## **🚀 Getting Started**

Follow these simple steps to get the project up and running on your local machine.

### **Prerequisites**

* Ensure you have **Python 3.x** installed.  
* **Strongly recommended:** Use a virtual environment to isolate project dependencies.

### **Installation**

1. **Clone the repository:**  
   ```
   git clone https://github.com/HayQacz/Movie-Recommender.git
   cd Movie-Recommender
   
2. **Create and activate a virtual environment:**
    ```
    python -m venv venv
    # On Windows:
   ```
   ```
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3. **Install the required Python packages:**  
    ```
    pip install \-r requirements.txt

### **Data Download**

1. Download the ml-1m.zip file from the official [MovieLens 1M Dataset page](https://grouplens.org/datasets/movielens/1m/).  
2. Extract the contents of the downloaded ml-1m.zip file. This will create a folder named ml-1m containing movies.dat, ratings.dat, users.dat, etc.  
3. Place this entire ml-1m folder into the data/raw/ directory within your cloned project. The final path should be movie\_recommendation\_system/data/raw/ml-1m/.

### **Running the Notebooks**

1. **Launch Jupyter Notebook:**  
   jupyter notebook

2. In the Jupyter interface, navigate to the notebooks/ directory.  
3. Execute the notebooks sequentially to follow the project's progression:  
   * 01\_data\_exploration\_preprocessing.ipynb  
   * 02\_model\_training\_evaluation.ipynb

## **📚 Notebooks Overview**

### **01\_data\_exploration\_preprocessing.ipynb**

This foundational notebook guides you through the initial phases of data handling:

* **Data Loading:** Efficiently loading the raw users.dat, movies.dat, and ratings.dat files.  
* **Exploratory Data Analysis (EDA):** A deep dive into the dataset's characteristics, including missing value checks, understanding rating and demographic distributions, and identifying popular movies.  
* **Data Merging:** Consolidating user, movie, and rating information into a unified, easy-to-use DataFrame.  
* **Feature Engineering:** Transforming the Genres column into a machine learning-ready one-hot encoded format.  
* **Data Saving:** Persisting the cleaned and processed DataFrames (merged\_movie\_data.csv and movies\_encoded.csv) to the data/processed/ directory for streamlined access in subsequent steps.

### **02\_model\_training\_evaluation.ipynb**

This notebook is the heart of our recommendation engine, focusing on model development and assessment:

* [ ] **Popularity-Based Recommender:** Implements a simple baseline model that suggests movies based on overall popularity.  
* [ ] **Content-Based Recommender (Genre-Based):** Develops a personalized recommender that suggests movies similar in genre to those a user has previously enjoyed, utilizing cosine similarity.  
* [ ] **Collaborative Filtering (Item-Based):** Explores a more advanced approach that recommends movies based on the similarity of user rating patterns between items.  
* [ ] **Model Evaluation Concepts:** Introduces and discusses key metrics for evaluating recommender systems, including RMSE/MAE for rating prediction and Precision@K/Recall@K for top-N recommendations, accompanied by a basic demonstration of data splitting.

## **✨ Recommendation Models Implemented**

This project showcases three distinct recommendation paradigms:

1. **Popularity-Based Recommender:**  
   * **Concept:** Recommends movies that have garnered the highest average ratings and a substantial number of total ratings across all users.  
   * **Pros:** Straightforward to implement, highly effective for addressing the "cold-start" problem (recommending to new users or suggesting new items).  
   * **Cons:** Lacks any form of personalization, offering the same recommendations to all users.  
2. **Content-Based Recommender (Genre-Based):**  
   * **Concept:** Constructs a unique profile for each user by analyzing the genres of movies they have rated highly. It then recommends unrated movies whose genre characteristics closely match the user's learned profile.  
   * **Pros:** Capable of recommending new or niche items, offers clear explainability for recommendations (e.g., "because you liked action movies").  
   * **Cons:** Limited by the available features of items, can lead to over-specialization (recommending only very similar items) if not carefully tuned.  
3. **Collaborative Filtering (Item-Based):**  
   * **Concept:** Identifies relationships between movies based on how users have rated them. If users who liked movie A also tended to like movie B, then A and B are considered similar. If a user likes A, B will be recommended.  
   * **Pros:** Uncovers complex, non-obvious patterns in user preferences, does not require explicit item features.  
   * **Cons:** Suffers from the "cold-start" problem for new users or items with no rating history, can face scalability challenges with extremely large datasets.

## **➡️ Next Steps & Future Enhancements**

This project is a solid foundation, and there are many exciting avenues for further development:

* **Refining Recommendation Logic:** Modularizing recommendation functions into dedicated Python scripts within the src/ directory for better code organization and reusability.  
* **Advanced Collaborative Filtering:** Implementing more sophisticated techniques such as Matrix Factorization (e.g., Singular Value Decomposition (SVD) or Non-negative Matrix Factorization (NMF)) using libraries like surprise for improved rating prediction and top-N recommendation accuracy.  
* **Hybrid Recommender Systems:** Combining the strengths of content-based and collaborative filtering approaches to mitigate their individual weaknesses and enhance overall recommendation quality.  
* **Interactive Interface:** Developing a user-friendly web application (e.g., using Streamlit or Flask) to allow real-time interaction with the recommendation system.  
* **Deployment:** Exploring strategies and platforms for deploying the recommendation system to make it accessible and operational.
