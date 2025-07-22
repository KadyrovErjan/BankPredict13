from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel


bank_app = FastAPI()

model = joblib.load('Bank_model.pkl')
scaler = joblib.load('scaler.pkl')


class PersonSchema(BaseModel):
    person_age: float
    person_income: float
    person_emp_exp: int
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    credit_score: float
    person_home_ownership: str
    person_gender: str
    person_education: str
    loan_intent: str
    previous_loan_defaults_on_file: str


@bank_app.post('/predict/')
async def predict(person: PersonSchema):
    print(person)
    person_dict = person.dict()
    print(person_dict)






    new_ownership = person_dict.pop('person_home_ownership')

    owner1_or_0 = [
        1 if new_ownership == 'OTHER' else 0,
        1 if new_ownership == 'OWN' else 0,
        1 if new_ownership == 'RENT' else 0,
    ]

    new_gender = person_dict.pop('person_gender')

    gender1_or_0 = [
        1 if new_gender == 'male' else 0
    ]

    new_education = person_dict.pop('person_education')

    education1_or_0 = [
        1 if new_education == 'Bachelor' else 0,
        1 if new_education == 'Doctorate' else 0,
        1 if new_education == 'High School' else 0,
        1 if new_education == 'Master' else 0,
    ]

    new_loan_intent = person_dict.pop('loan_intent')

    loan_intent1_or_0 = [
        1 if new_loan_intent == 'EDUCATION' else 0,
        1 if new_loan_intent == 'HOMEIMPROVEMENT' else 0,
        1 if new_loan_intent == 'MEDICAL' else 0,
        1 if new_loan_intent == 'PERSONAL' else 0,
        1 if new_loan_intent == 'VENTURE' else 0,
    ]

    new_previous_loan_defaults_on_file= person_dict.pop('previous_loan_defaults_on_file')

    previous_loan_defaults_on_file1_or_0 = [
        1 if new_previous_loan_defaults_on_file == 'Yes' else 0
    ]

    features = list(person_dict.values()) + owner1_or_0 + gender1_or_0 + education1_or_0 + loan_intent1_or_0 + previous_loan_defaults_on_file1_or_0
    print(features)

    scaled = scaler.transform([features])
    print(model.predict(scaled))
    print(model.predict(scaled)[0])
    pred = model.predict(scaled)[0]
    print(model.predict_proba(scaled))
    print(model.predict_proba(scaled)[0][1])

    prob = model.predict_proba(scaled)[0][1]

    return {"approved": bool(pred), "probability": round(prob, 2)}

if __name__ == '__main__':
    uvicorn.run(bank_app, host="127.0.0.1", port=8001)