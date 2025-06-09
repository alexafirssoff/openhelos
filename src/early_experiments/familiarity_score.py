import math


def calculate_familiarity_score(F, k=0.7) -> float:
    """Calculates the Familiarity Score (FS) from Free Energy (F)."""
    # Check F before calculations
    if not isinstance(F, (int, float)) or F < 0 or math.isnan(F) or math.isinf(F):
        return 0.0
    try:
        # Use the passed k
        fs = math.exp(-k * F)
        # Check the result for validity (although exp usually does not produce nan/inf from float >= 0)
        if math.isnan(fs) or math.isinf(fs):
            return 0.0
        return fs
    except OverflowError:
        # The exponent of a very large negative number (F -> inf) should be 0
        return 0.0
    except Exception:
        # Other possible math.exp errors (although unlikely)
        return 0.0


if __name__ == '__main__':
    # Example of use:
    F_low = 0.1  # Almost no surprise
    F_medium = 1.5  # Moderate surprise
    F_high = 23.0  # Strong surprise
    
    k_factor = 1.0  # Sensitivity coefficient
    
    fs_low = calculate_familiarity_score(F_low, k_factor)
    fs_medium = calculate_familiarity_score(F_medium, k_factor)
    fs_high = calculate_familiarity_score(F_high, k_factor)
    
    print(f"F={F_low}, FS={fs_low:.4f}")  # Expected close to 1
    print(f"F={F_medium}, FS={fs_medium:.4f}")  # An intermediate value is expected
    print(f"F={F_high}, FS={fs_high:.4f}")  # Expected close to 0