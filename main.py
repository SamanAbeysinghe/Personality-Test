from flask import Flask, request, jsonify
import joblib
import numpy as np

main = Flask(__name__)

model = joblib.load('personality_classifier_RF.pkl')
scaler = joblib.load('scaler_RF.pkl')


@main.route('/', methods=['GET'])
def index():
    return '''
            <h2>Personality Predictor</h2>
            <form method="post" action="/predict_form">
                Time Spent Alone (0-10): <input type="number" name="Time_spent_Alone"><br><br>
                Stage Fear (1 = Yes, 0 = No): <input type="number" name="Stage_fear"><br><br>
                Social Event Attendance (0-10): <input type="number" name="Social_event_attendance"><br><br>
                Going Outside (0-7): <input type="number" name="Going_outside"><br><br>
                Drained After Socializing (1 = Yes, 0 = No): <input type="number" name="Drained_after_socializing"><br><br>
                Friends Circle Size: <input type="number" name="Friends_circle_size"><br><br>
                Post Frequency (0-10): <input type="number" name="Post_frequency"><br><br>
                <input type="submit" value="Predict Personality">
            </form>
            '''


@main.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        input_data = np.array([[
            int(request.form['Time_spent_Alone']),
            int(request.form['Stage_fear']),
            int(request.form['Social_event_attendance']),
            int(request.form['Going_outside']),
            int(request.form['Drained_after_socializing']),
            int(request.form['Friends_circle_size']),
            int(request.form['Post_frequency'])
        ]])

        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]
        result = 'Extrovert' if prediction == 1 else 'Introvert'

        return f"<h3>Predicted Personality: {result}</h3><a href='/'>Go Back</a>"

    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><a href='/'>Go Back</a>"


@main.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()

    try:
        input_data = np.array([[
            data['Time_spent_Alone'],
            int(data['Stage_fear']), data['Social_event_attendance'],
            data['Going_outside'],
            int(data['Drained_after_socializing']),
            data['Friends_circle_size'], data['Post_frequency']
        ]])

        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]
        result = 'Extrovert' if prediction == 0 else 'Introvert'

        return jsonify({'personality': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    main.run(debug=True, host='0.0.0.0')
