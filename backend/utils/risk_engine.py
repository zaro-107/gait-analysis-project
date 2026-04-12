def calculate_risk(confidence, step_length, cadence, symmetry):
    risk = 0
    
    # Model confidence (50%)
    risk += confidence * 50
    
    # Step length (short = risky)
    if step_length < 0.8:
        risk += 15
        
    # Cadence (slow = risky)
    if cadence < 60:
        risk += 15
        
    # Symmetry imbalance
    if symmetry > 0.05:
        risk += 20
        
    return min(risk, 100)


def get_severity(risk):
    if risk >= 75:
        return "High"
    elif risk >= 50:
        return "Moderate"
    else:
        return "Low"


def doctor_recommendation(severity):
    if severity == "High":
        return [
            "Immediate consultation with a neurologist",
            "MRI or brain scan may be required",
            "Start physiotherapy assessment"
        ]
    elif severity == "Moderate":
        return [
            "Consult a doctor soon",
            "Monitor walking pattern",
            "Start balance exercises"
        ]
    else:
        return [
            "Maintain healthy lifestyle",
            "Regular walking exercise",
            "Re-test after few weeks"
        ]