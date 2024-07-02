% Background knowledge
% Basic predicates
is_male(male).
is_female(female).

% Age categories
is_adult(Age) :- Age >= 18.
is_senior(Age) :- Age >= 65.
is_middle_aged(Age) :- Age >= 45, Age < 65.
is_young_adult(Age) :- Age >= 18, Age < 45.

% Health risk factors
high_glucose(Glucose) :- Glucose > 180.
normal_glucose(Glucose) :- Glucose =< 180, Glucose >= 70.
low_glucose(Glucose) :- Glucose < 70.

% BMI categories
underweight(BMI) :- BMI < 18.5.
normal_weight(BMI) :- BMI >= 18.5, BMI < 24.9.
overweight(BMI) :- BMI >= 25, BMI < 29.9.
obesity(BMI) :- BMI >= 30.

% Smoking status
is_smoker(smokes).
is_non_smoker(never_smoked).
is_former_smoker(formerly_smoked).
is_unknown_smoker(unknown).

% Other predicates
has_hypertension(H) :- H = 1.
has_heart_disease(HD) :- HD = 1.
is_married(yes).
is_not_married(no).
