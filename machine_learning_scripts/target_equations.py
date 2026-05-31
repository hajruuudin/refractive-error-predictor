import numpy as np

class TargetEquations:
    def __init__(self):
        pass

    def calculate_weighted_average(self, normalized_df):
        myopia_score = normalized_df["myopia_score"] = (
            (normalized_df["myopia_level"]) * 0.25 + (normalized_df["refractive_worsening"]) * 0.75
        )

        computervs_score = normalized_df["cvs_score"] = (
            (normalized_df["cvs_headache_strain"]) * 0.50 + (normalized_df["cvs_dry_eyes"]) * 0.50
        )

        astigmatism_score = normalized_df["astigmatism_score"] = (
            (normalized_df["astigmatism_symptoms"]) * 0.30 + (normalized_df["myopia_level"]) * 0.70
        )

        return (myopia_score, computervs_score, astigmatism_score)
    
    def calculate_power_mean(self, normalized_df):
        p = 2
        
        myopia_score = normalized_df["myopia_score"] = (
            ((normalized_df["myopia_level"]**p + normalized_df["refractive_worsening"]**p) / 2) ** (1/p)
        )

        computervs_score = normalized_df["cvs_score"] = (
            ((normalized_df["cvs_headache_strain"]**p + normalized_df["cvs_dry_eyes"]**p) / 2) ** (1/p)
        )

        astigmatism_score = normalized_df["astigmatism_score"] = (
            ((normalized_df["astigmatism_symptoms"]**p + normalized_df["myopia_level"]**p) / 2) ** (1/p)
        )

        return (myopia_score, computervs_score, astigmatism_score)

    def calculate_geometric_mean(self, normalized_df):
        myopia_score = normalized_df["myopia_score"] = np.sqrt(
            (normalized_df["myopia_level"]) * (normalized_df["refractive_worsening"] + 1e-6)
        )

        computervs_score = normalized_df["cvs_score"] = np.sqrt(
            (normalized_df["cvs_headache_strain"]) * (normalized_df["cvs_dry_eyes"] + 1e-6)
        )

        astigmatism_score = normalized_df["astigmatism_score"] = np.sqrt(
            (normalized_df["astigmatism_symptoms"]) * (normalized_df["myopia_level"] + 1e-6)
        )

        return (myopia_score, computervs_score, astigmatism_score)

    def calculate_normalized_sum(self, normalized_df):
        myopia_score = normalized_df["myopia_score"] = (
            (normalized_df["myopia_level"] + normalized_df["refractive_worsening"]) / 2
        )

        computervs_score = normalized_df["cvs_score"] = (
            (normalized_df["cvs_headache_strain"] + normalized_df["cvs_dry_eyes"]) / 2
        )

        astigmatism_score = normalized_df["astigmatism_score"] = (
            (normalized_df["astigmatism_symptoms"] + normalized_df["myopia_level"]) / 2
        )

        return (myopia_score, computervs_score, astigmatism_score)

    def calculate_max_penalty(self, normalized_df):
        myopia_score = normalized_df["myopia_score"] = np.maximum(
            normalized_df["myopia_level"], normalized_df["refractive_worsening"]
        ) * (1 - 0.2 * (1 - np.minimum(normalized_df["myopia_level"], normalized_df["refractive_worsening"])))

        computervs_score = normalized_df["cvs_score"] = np.maximum(
            normalized_df["cvs_headache_strain"], normalized_df["cvs_dry_eyes"]
        )

        astigmatism_score = normalized_df["astigmatism_score"] = np.where(
            normalized_df["myopia_level"] == 0,
            normalized_df["astigmatism_symptoms"] * 0.5,
            normalized_df["astigmatism_symptoms"]
        )

        return (myopia_score, computervs_score, astigmatism_score)

    def calculate_harmonic_mean(self, normalized_df):
        myopia_score = normalized_df["myopia_score"] = 2 / (
            1 / (normalized_df["myopia_level"] + 1e-6) + 1 / (normalized_df["refractive_worsening"] + 1e-6)
        )

        computervs_score = normalized_df["cvs_score"] = 2 / (
            1 / (normalized_df["cvs_headache_strain"] + 1e-6) + 1 / (normalized_df["cvs_dry_eyes"] + 1e-6)
        )

        astigmatism_score = normalized_df["astigmatism_score"] = 2 / (
            1 / (normalized_df["astigmatism_symptoms"] + 1e-6) + 1 / (normalized_df["myopia_level"] + 1e-6)
        )

        return (myopia_score, computervs_score, astigmatism_score)


    
