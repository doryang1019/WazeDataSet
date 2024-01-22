The dataset “Waze User Retained vs. Churned” contains data collected from Waze app users for a month. It has 1 Id
(correlative number), 1 label and 11 features:

Label:
1. Label: Binary target variable (“retained” vs “churned”) for if a user has churned anytime during the month.
Features:
1. Sessions: Times a user opened the app during the month.
2. Drives: Times a user drove at least 1 km during the month.
3. Total_sessions: A model estimate of the total number of sessions since a user has onboarded.
4. N_days_after_onboarding: The number of days since a user signed up for the app.
5. Total_navigations_fav1: Total navigations since onboarding to the user’s favorite place.
6. Total_navigations_fav2: Total navigations since onboarding to the user’s second favorite place.
7. Driven_km_drives: Total kilometers driven during the month.
8. Duration_minutes_drives: Total duration driven in minutes during the month.
9. Activity_days: Number of days the user opened the app during the month.
10. Driving_days: Number of days the user drove at least 1 km during the month.
11. Device: The type of device a user starts a session with.


The libraries used for elaborating charts are as follows:
• Numpy
• Pandas
• Matplotlib
• Seaborn
• Scipy
• Sklearn
• TensorFlow
