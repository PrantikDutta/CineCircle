# CineCircle — Group Movie Recommendation System

## Project Overview

Choosing a movie with a group can be surprisingly difficult because everyone has different preferences. CineCircle was developed to address this problem by creating a **group movie recommendation system** that combines the preferences of multiple users to recommend movies that the group may enjoy.

Instead of generating recommendations based on a single user's preferences, CineCircle allows multiple friends to enter their favourite movies and combines their preferences using a **content-based recommendation approach**.

The application is built using Python and machine learning techniques, deployed using Streamlit, and integrated with the TMDB API to provide additional movie information.

---

## Objective

The primary objective of CineCircle is to develop a simple and interactive recommendation system that can:

* Accept favourite movies from multiple users.
* Combine the preferences of different users.
* Identify movies similar to the group's selected movies.
* Generate recommendations based on the collective preferences.
* Display movie posters and additional information through the TMDB API.
* Provide an accessible web interface through Streamlit.

---

## How CineCircle Works

The recommendation pipeline can be summarized as:

```text
User 1 Favourite Movies
          \
User 2 Favourite Movies
           \
User 3 Favourite Movies
             ↓
     Combine Preferences
             ↓
       Movie Features
             ↓
         TF-IDF
             ↓
   Cosine Similarity Matrix
             ↓
    Group Recommendation
             ↓
      TMDB Movie Details
             ↓
      Recommended Movies
```

---

## Recommendation Methodology

### 1. User Preference Input

CineCircle allows multiple friends to enter their favourite movies.

The number of participants can be selected within the application, allowing the system to adapt to different group sizes.

For example:

```text
Friend 1 → Interstellar, Inception
Friend 2 → The Dark Knight, Joker
Friend 3 → Arrival, Interstellar
```

The system then combines these preferences to generate recommendations for the group.

---

### 2. Movie Feature Representation

Movie information is converted into a text-based representation using relevant movie attributes.

The project uses features such as:

* Movie title
* Genres
* Keywords/tags
* Other available descriptive information

These features are combined into a representation that captures the characteristics of each movie.

---

### 3. TF-IDF Vectorization

**TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert the movie descriptions into numerical vectors.

The basic idea is to assign higher importance to terms that are informative for a particular movie while reducing the importance of very common terms.

This allows movies to be represented mathematically based on their content.

---

### 4. Cosine Similarity

Once the movies are represented as TF-IDF vectors, **cosine similarity** is used to measure the similarity between movies.

The cosine similarity between two vectors can be represented as:

$$
\text{Cosine Similarity}(A,B)
=
\frac{A \cdot B}{||A||\,||B||}
$$

A higher similarity score indicates that two movies have more similar content characteristics.

---

### 5. Group Recommendation

Instead of considering only one user's preferences, CineCircle considers the favourite movies entered by multiple users.

The similarity scores associated with these movies are combined to identify recommendations that better represent the group's collective preferences.

This makes the recommendation process more suitable for situations where several people need to agree on a movie.

---

## Features

### Group Recommendations

Users can enter favourite movies for multiple friends and receive recommendations based on the group's combined preferences.

### Dynamic Group Size

The application allows users to select the number of friends participating in the recommendation.

For mobile users, the group-size slider can be accessed through the top-right `>>` menu.

### Movie Information

The TMDB API is integrated to retrieve information such as:

* Movie posters
* Ratings
* Movie details

### Interactive Web Application

The application is deployed using Streamlit, allowing users to interact with the recommendation system directly through a web browser.

---

## Technologies Used

* **Python**
* **Pandas** — Data manipulation
* **NumPy** — Numerical computation
* **Scikit-learn** — TF-IDF vectorization and cosine similarity
* **Streamlit** — Web application and deployment
* **TMDB API** — Movie information and posters
* **Jupyter Notebook** — Data analysis and experimentation

---

## Dataset

The recommendation system uses a movie dataset containing information such as:

* Movie ID
* Movie title
* Tags
* Vote average
* Runtime
* Popularity

The movie descriptions/tags are used as the primary information for the content-based recommendation system.

The dataset is relatively old, which is one of the current limitations of the application.

---

## Project Structure

```text
CineCircle/
│
├── app.py
├── movies.pkl
├── vectors.pkl
├── similarity.pkl
├── tfidf.pkl
│
├── requirements.txt
└── README.md
```

### Important Files

| File               | Description                          |
| ------------------ | ------------------------------------ |
| `app.py`           | Main Streamlit application           |
| `movies.pkl`       | Processed movie dataset              |
| `vectors.pkl`      | TF-IDF movie vectors                 |
| `similarity.pkl`   | Precomputed cosine similarity matrix |
| `tfidf.pkl`        | TF-IDF vectorizer                    |
| `requirements.txt` | Required Python packages             |

---

## Deployment

CineCircle is deployed using **Streamlit** and can be accessed through the live application:

**Live Application:**
https://cinecircle-movie.streamlit.app/

The application uses the TMDB API to dynamically retrieve movie information and posters.

---

## Limitations

CineCircle is intentionally built as a relatively simple content-based recommendation system, and it has several limitations.

### Older Dataset

The dataset used by the system is not completely up to date. As a result, newer movies may not be represented in the recommendations.

### Content-Based Recommendations

The system primarily relies on similarity between movie features. It does not use advanced collaborative filtering, deep learning, or generative AI techniques.

### Franchise Recommendations

Movies belonging to the same franchise can sometimes produce questionable recommendations. For example, when users select movies from franchises such as **Avengers** or **Harry Potter**, the similarity-based approach may overemphasize shared terms or franchise-related characteristics.

### Limited Understanding of User Context

The system does not currently understand more complex preferences such as:

* "Something like Interstellar but less serious"
* "A comedy that everyone can watch"
* "Something similar to this movie but shorter"

The current system primarily relies on measurable similarity between movie features.

---

## What I Learned

This project provided practical experience in developing a machine learning application from the initial recommendation logic to deployment.

Key areas of learning included:

* Building a content-based recommendation system.
* Applying TF-IDF to real-world text data.
* Understanding cosine similarity.
* Combining multiple users' preferences.
* Working with APIs.
* Building an interactive application using Streamlit.
* Deploying a machine learning application.
* Managing precomputed model/data files.
* Understanding the limitations of similarity-based recommendation systems.

One of the most valuable aspects of the project was not just building the recommendation model, but testing it and identifying situations where the model produces unexpected recommendations.

---

## Future Improvements

Several improvements could make CineCircle more robust:

* Incorporating newer movie datasets.
* Combining content-based and collaborative filtering approaches.
* Introducing user ratings instead of relying only on favourite movies.
* Using genre-level preference weighting.
* Incorporating popularity and rating information into recommendations.
* Improving franchise handling.
* Using more advanced recommendation algorithms.
* Allowing users to specify preferences such as genre, language, runtime, or release year.
* Developing a hybrid recommendation system combining multiple sources of information.

---

## Conclusion

CineCircle demonstrates how a relatively simple machine learning approach can be used to solve a practical recommendation problem.

The core idea is:

```text
Multiple User Preferences
          ↓
   Feature Extraction
          ↓
      TF-IDF
          ↓
 Cosine Similarity
          ↓
 Group Recommendations
```

Although the current system has limitations, building and deploying CineCircle provided hands-on experience with the complete machine learning workflow — from data processing and feature engineering to model implementation, API integration, deployment, and evaluation.

---

## Author

**Prantik Dutta**
M.Sc. Statistics and Computing
Banaras Hindu University (BHU)

---

## Topics

`Python` `Machine Learning` `Recommendation Systems` `TF-IDF` `Cosine Similarity` `Streamlit` `Data Science` `TMDB API` `NLP` `Content-Based Filtering`
