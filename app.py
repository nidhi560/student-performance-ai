from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)


# ============================================================
# 1. SAMPLE DATASET
# ============================================================

data = {
    "study_hours": [
        1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
        6, 7, 7, 8, 8, 2, 3, 4, 5, 6
    ],

    "attendance": [
        50, 55, 60, 65, 68, 70, 75, 78, 80, 82,
        85, 88, 90, 92, 95, 62, 70, 76, 84, 89
    ],

    "previous_marks": [
        40, 45, 50, 55, 58, 60, 65, 70, 72, 75,
        78, 82, 85, 88, 92, 48, 57, 66, 74, 80
    ],

    "assignment_score": [
        45, 50, 55, 60, 62, 65, 70, 74, 76, 80,
        82, 85, 88, 92, 95, 52, 60, 68, 78, 86
    ],

    "sleep_hours": [
        5, 5, 6, 6, 6, 7, 7, 7, 8, 8,
        8, 8, 8, 9, 9, 6, 7, 7, 8, 8
    ],

    "extracurricular": [
        0, 0, 1, 0, 1, 0, 1, 1, 0, 1,
        1, 0, 1, 1, 0, 1, 0, 1, 0, 1
    ],

    "final_score": [
        42, 47, 52, 57, 60, 64, 69, 74, 77, 81,
        84, 88, 91, 94, 97, 50, 59, 68, 79, 86
    ]
}


df = pd.DataFrame(data)


# ============================================================
# 2. MACHINE LEARNING MODEL
# ============================================================

features = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score",
    "sleep_hours",
    "extracurricular"
]

X = df[features]
y = df["final_score"]


model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)


# ============================================================
# 3. PERFORMANCE FUNCTION
# ============================================================

def get_performance(score):

    if score >= 90:
        return "Excellent", "Outstanding performance!"

    elif score >= 75:
        return "Very Good", "Very good academic performance."

    elif score >= 60:
        return "Good", "Good performance. Keep improving."

    elif score >= 50:
        return "Average", "Average performance. More practice is recommended."

    else:
        return "Needs Improvement", "Student needs academic improvement."


# ============================================================
# 4. HTML PAGE
# ============================================================

HTML = """

<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>Student Performance AI</title>


<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family: Arial, sans-serif;

    background: linear-gradient(
        135deg,
        #667eea,
        #764ba2
    );

    min-height: 100vh;

    padding: 30px;

}


.container {

    max-width: 900px;

    margin: auto;

}


.header {

    text-align: center;

    color: white;

    margin-bottom: 25px;

}


.header h1 {

    font-size: 38px;

    margin-bottom: 10px;

}


.header p {

    font-size: 17px;

}


.card {

    background: white;

    padding: 30px;

    border-radius: 20px;

    box-shadow:
        0 15px 40px
        rgba(0,0,0,0.2);

}


.form-grid {

    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 20px;

}


.form-group {

    display: flex;

    flex-direction: column;

}


label {

    font-weight: bold;

    margin-bottom: 8px;

}


input,
select {

    padding: 13px;

    border: 1px solid #ddd;

    border-radius: 8px;

    font-size: 15px;

}


button {

    width: 100%;

    margin-top: 25px;

    padding: 15px;

    border: none;

    border-radius: 10px;

    background: #667eea;

    color: white;

    font-size: 18px;

    font-weight: bold;

    cursor: pointer;

}


button:hover {

    background: #5369d8;

}


.result {

    margin-top: 30px;

    padding: 25px;

    border-radius: 15px;

    text-align: center;

    background: #f5f7ff;

}


.score {

    font-size: 55px;

    font-weight: bold;

    color: #667eea;

}


.category {

    font-size: 25px;

    font-weight: bold;

    margin: 10px;

}


.info {

    color: #555;

}


.error {

    margin-top: 20px;

    padding: 15px;

    background: #ffe5e5;

    color: #c00;

    border-radius: 8px;

}


.footer {

    text-align: center;

    color: white;

    margin-top: 20px;

}


@media(max-width:650px) {

    .form-grid {

        grid-template-columns: 1fr;

    }

    .header h1 {

        font-size: 28px;

    }

    body {

        padding: 15px;

    }

}

</style>

</head>


<body>


<div class="container">


<div class="header">

<h1>🎓 Student Performance AI</h1>

<p>
AI Based Student Performance Prediction System
</p>

</div>


<div class="card">


<form method="POST">


<div class="form-grid">


<div class="form-group">

<label>
Study Hours Per Day
</label>

<input
type="number"
name="study_hours"
min="0"
max="24"
step="0.1"
placeholder="Example: 5"
required
>

</div>


<div class="form-group">

<label>
Attendance (%)
</label>

<input
type="number"
name="attendance"
min="0"
max="100"
step="0.1"
placeholder="Example: 85"
required
>

</div>


<div class="form-group">

<label>
Previous Marks (%)
</label>

<input
type="number"
name="previous_marks"
min="0"
max="100"
step="0.1"
placeholder="Example: 75"
required
>

</div>


<div class="form-group">

<label>
Assignment Score (%)
</label>

<input
type="number"
name="assignment_score"
min="0"
max="100"
step="0.1"
placeholder="Example: 80"
required
>

</div>


<div class="form-group">

<label>
Sleep Hours Per Day
</label>

<input
type="number"
name="sleep_hours"
min="0"
max="24"
step="0.1"
placeholder="Example: 8"
required
>

</div>


<div class="form-group">

<label>
Extracurricular Activity
</label>

<select name="extracurricular">

<option value="1">
Yes
</option>

<option value="0">
No
</option>

</select>

</div>


</div>


<button type="submit">

🤖 Predict Student Performance

</button>


</form>


{% if prediction %}

<div class="result">

<h2>AI Prediction Result</h2>


<div class="score">

{{ prediction }}/100

</div>


<div class="category">

{{ category }}

</div>


<p class="info">

{{ message }}

</p>


</div>

{% endif %}


{% if error %}

<div class="error">

{{ error }}

</div>

{% endif %}


</div>


<div class="footer">

<p>
Student Performance Prediction using
Artificial Intelligence & Machine Learning
</p>

</div>


</div>


</body>

</html>

"""


# ============================================================
# 5. FLASK ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])

def home():

    prediction = None

    category = None

    message = None

    error = None


    if request.method == "POST":

        try:

            study_hours = float(
                request.form["study_hours"]
            )

            attendance = float(
                request.form["attendance"]
            )

            previous_marks = float(
                request.form["previous_marks"]
            )

            assignment_score = float(
                request.form["assignment_score"]
            )

            sleep_hours = float(
                request.form["sleep_hours"]
            )

            extracurricular = int(
                request.form["extracurricular"]
            )


            # Input validation

            if not 0 <= attendance <= 100:

                raise ValueError(
                    "Attendance must be between 0 and 100."
                )


            if not 0 <= previous_marks <= 100:

                raise ValueError(
                    "Previous marks must be between 0 and 100."
                )


            if not 0 <= assignment_score <= 100:

                raise ValueError(
                    "Assignment score must be between 0 and 100."
                )


            # Create input DataFrame

            student = pd.DataFrame(
                [[
                    study_hours,
                    attendance,
                    previous_marks,
                    assignment_score,
                    sleep_hours,
                    extracurricular
                ]],
                columns=features
            )


            # AI Prediction

            prediction = model.predict(student)[0]


            # Keep score between 0 and 100

            prediction = max(
                0,
                min(100, prediction)
            )


            prediction = round(
                prediction,
                2
            )


            # Performance category

            category, message = get_performance(
                prediction
            )


        except Exception as e:

            error = str(e)


    return render_template_string(

        HTML,

        prediction=prediction,

        category=category,

        message=message,

        error=error

    )


# ============================================================
# 6. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )