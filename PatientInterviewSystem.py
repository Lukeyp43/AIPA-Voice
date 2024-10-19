class PatientInterviewSystem:
    def __init__(self):
        self.patient_data = {}
        
    def get_chief_complaint(self):
        return {
            "question": "What brings you in today? What's your main concern?",
            "type": "open",
            "next": self.assess_symptom_duration
        }
        
    def assess_symptom_duration(self, chief_complaint):
        self.patient_data['chief_complaint'] = chief_complaint
        return {
            "question": "When did these symptoms begin?",
            "type": "temporal",
            "next": self.symptom_characteristics,
            "follow_up": {
                "acute": "Did this start suddenly or gradually?",
                "chronic": "Has this been constant or does it come and go?"
            }
        }
    
    def symptom_characteristics(self, duration):
        self.patient_data['duration'] = duration
        symptoms_questions = {
            "pain": {
                "questions": [
                    "On a scale of 1-10, how severe is the pain?",
                    "Can you describe the pain? (Sharp, dull, throbbing, etc.)",
                    "What makes it better or worse?",
                    "Does it radiate anywhere?"
                ],
                "next": self.pain_assessment
            },
            "respiratory": {
                "questions": [
                    "Are you experiencing shortness of breath?",
                    "Any cough? If yes, is it productive?",
                    "Any chest pain with breathing?"
                ],
                "next": self.respiratory_assessment
            },
            "gastrointestinal": {
                "questions": [
                    "Have you experienced any nausea or vomiting?",
                    "Any changes in appetite?",
                    "Any changes in bowel movements?"
                ],
                "next": self.gi_assessment
            }
        }
        return symptoms_questions.get(self.categorize_complaint())
    
    def pain_assessment(self, responses):
        self.patient_data['pain_characteristics'] = responses
        return {
            "questions": [
                "Have you tried any medications for the pain?",
                "What helps relieve the pain?",
                "Does the pain affect your daily activities?",
                "Have you had similar pain before?"
            ],
            "next": self.review_of_systems
        }
    
    def respiratory_assessment(self, responses):
        self.patient_data['respiratory_symptoms'] = responses
        return {
            "questions": [
                "Any fever?",
                "Have you been exposed to anyone who's sick?",
                "Do you have any history of asthma or COPD?",
                "Are you a smoker or former smoker?"
            ],
            "next": self.review_of_systems
        }
    
    def gi_assessment(self, responses):
        self.patient_data['gi_symptoms'] = responses
        return {
            "questions": [
                "Any recent changes in diet?",
                "Any blood in stool?",
                "Any recent weight changes?",
                "Any history of similar problems?"
            ],
            "next": self.review_of_systems
        }
    
    def review_of_systems(self, specific_assessment):
        self.patient_data['specific_assessment'] = specific_assessment
        return {
            "systems": {
                "constitutional": [
                    "Fever",
                    "Fatigue",
                    "Weight changes"
                ],
                "cardiovascular": [
                    "Chest pain",
                    "Palpitations",
                    "Edema"
                ],
                "neurological": [
                    "Headaches",
                    "Dizziness",
                    "Numbness/tingling"
                ]
            },
            "next": self.past_medical_history
        }
    
    def past_medical_history(self, ros_responses):
        self.patient_data['review_of_systems'] = ros_responses
        return {
            "questions": [
                "Do you have any chronic medical conditions?",
                "Have you had any surgeries?",
                "Are you currently taking any medications?",
                "Do you have any allergies to medications?",
                "Is there any relevant family history?",
                "Do you smoke, drink alcohol, or use any recreational drugs?"
            ],
            "next": self.social_history
        }
    
    def social_history(self, pmh_responses):
        self.patient_data['medical_history'] = pmh_responses
        return {
            "questions": [
                "What is your occupation?",
                "Who do you live with?",
                "Do you feel safe at home?",
                "Any recent travel?",
                "Any major stressors in your life currently?"
            ],
            "next": self.generate_assessment
        }
    
    def categorize_complaint(self):
        # Logic to categorize the chief complaint into appropriate category
        # This would analyze the chief_complaint text and return appropriate category
        # Example implementation:
        complaint = self.patient_data['chief_complaint'].lower()
        if any(word in complaint for word in ['pain', 'ache', 'hurt', 'sore']):
            return 'pain'
        elif any(word in complaint for word in ['breath', 'cough', 'chest']):
            return 'respiratory'
        elif any(word in complaint for word in ['stomach', 'nausea', 'vomit', 'diarrhea']):
            return 'gastrointestinal'
        return 'general'
    
    def generate_assessment(self, social_responses):
        self.patient_data['social_history'] = social_responses
        # Generate preliminary assessment and plan based on collected data
        return {
            "assessment": "Based on collected information...",
            "plan": [
                "Recommended tests",
                "Treatment options",
                "Follow-up plans",
                "Patient education points"
            ]
        }

"""
# Example usage
def main():
    interview = PatientInterviewSystem()
    
    # Start interview
    current_step = interview.get_chief_complaint()
    
    # Example interaction flow
    responses = {
        "chief_complaint": "I've been having severe stomach pain",
        "duration": "Started 3 days ago",
        "characteristics": {
            "severity": "7/10",
            "description": "Sharp, cramping pain",
            "location": "Lower abdomen",
            "modifying_factors": "Worse after eating"
        }
    }
    
    # Process through decision tree
    next_step = current_step['next'](responses['chief_complaint'])
    # Continue through the rest of the interview steps...

if __name__ == "__main__":
    main()

"""